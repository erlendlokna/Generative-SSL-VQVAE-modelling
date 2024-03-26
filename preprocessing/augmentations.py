import numpy as np

import torch
import random
from scipy.ndimage import rotate, affine_transform
import torch.nn.functional as F
import scipy


class Augmenter(object):
    def __init__(self, time_augs, timefreq_augs, aug_params, use_all_methods, **kwargs):
        self.time_augs = time_augs if time_augs is not None else []
        self.timefreq_augs = timefreq_augs if timefreq_augs is not None else []

        self.use_all_method = use_all_methods

        self.time_augmenter = TimeAugmenter(**aug_params)
        self.timefreq_augmenter = TimeFreqAugmenter(**aug_params)

    def augment(self, input_timeseries, return_combinations=False):
        """
        Augments a single time series with a random combination of augmentation methods.
        """
        if isinstance(input_timeseries, torch.Tensor):
            input_timeseries = input_timeseries.numpy()
        elif not isinstance(input_timeseries, np.ndarray):
            raise ValueError(
                "input_timeseries should be a numpy array or a torch tensor"
            )

        X = input_timeseries.copy()

        # Randomly constructing a combination of augmentation methods
        if self.use_all_method:
            picked_augs = self.time_augs + self.timefreq_augs
            time_methods_to_apply = self.time_augs
            timefreq_methods_to_apply = self.timefreq_augs
        else:
            all_augs = self.time_augs + self.timefreq_augs
            np.random.shuffle(all_augs)
            picked_augs = all_augs[: np.random.randint(1, len(all_augs))]
            time_methods_to_apply = set(picked_augs).intersection(self.time_augs)
            timefreq_methods_to_apply = set(picked_augs).intersection(
                self.timefreq_augs
            )

        # Applying time augmentations
        X = self.time_augmenter.apply_augmentations(time_methods_to_apply, X)

        # Convert to time-frequency representation
        U = self.timefreq_augmenter.stft(X).numpy()

        # Applying time-frequency augmentations
        U = self.timefreq_augmenter.apply_augmentations(timefreq_methods_to_apply, U)

        # converting back
        X = self.timefreq_augmenter.istft(
            torch.from_numpy(U), original_length=X.shape[-1]
        )

        if return_combinations:
            return X, picked_augs
        else:
            return X


class TimeAugmenter(object):
    def __init__(
        self, AmpR_rate, slope_rate, noise_std, window_ratio, n_segments, **kwargs
    ):
        self.AmpR_rate = AmpR_rate
        self.slope_rate = slope_rate
        self.noise_std = noise_std
        self.window_ratio = window_ratio
        self.n_segments = n_segments

        # config method mapping:
        self.method_mapping = {
            "amplitude_resize": "add_amplitude_resize",
            "flip": "add_flip",
            "slope": "add_slope",
            "jitter": "add_jitter",
            "noise_window": "add_noise_window",
            "window_warp": "add_window_warp",
            "magnitude_warp": "add_magnitude_warp",
            "slice_and_shuffle": "add_slice_and_shuffle",
            "gaussian_noise": "add_gaussian_noise",
        }

    def apply_augmentation(self, method_name, input):
        if method_name in self.method_mapping:
            method = getattr(self, self.method_mapping[method_name])
            return method(input)
        else:
            raise ValueError(
                f"{method_name} is not a valid method for time augmentation"
            )

    def apply_augmentations(self, method_names, input):
        X = input.copy()
        for method_name in method_names:
            input = self.apply_augmentation(method_name, input)
        return input

    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # --- Augmentation methods ---
    def add_amplitude_resize(self, *subx_views):
        """
        Apply random amplitude resizing to input sequences.

        Parameters:
        - subx_views: Variable number of input sequences (subseq_len).

        Returns:
        - augmented_views: List of sequences with random amplitude resizing.
        """

        augmented_views = []

        for subx in subx_views:
            subseq_len = subx.shape[0]
            mul_AmpR = 1 + np.random.normal(0, self.AmpR_rate, size=(subseq_len,))
            augmented_view = subx * mul_AmpR

            augmented_views.append(augmented_view)

        if len(augmented_views) == 1:
            augmented_views = augmented_views[0]

        return augmented_views

    def add_flip(self, *subx_views):
        """
        Randomly flip the input sequences horizontally.
        """
        flipped_subx_views = [np.flip(subx, axis=-1) for subx in subx_views]

        if len(flipped_subx_views) == 1:
            flipped_subx_views = flipped_subx_views[0]

        return flipped_subx_views

    def add_gaussian_noise(self, *subx_views, mean=0, variance=0.01):
        """
        Add Gaussian noise to the input sequences.
        """
        noise_subx_views = []

        for subx in subx_views:
            noise = np.random.normal(mean, np.sqrt(variance), subx.shape)
            noise_subx = subx + noise
            noise_subx_views.append(noise_subx)

        if len(noise_subx_views) == 1:
            noise_subx_views = noise_subx_views[0]

        return noise_subx_views

    def add_slope(self, *subx_views):
        """
        Add a linear slope to the input sequences.
        """
        sloped_subx_views = []

        for subx in subx_views:
            # Handle 1D data
            subseq_len = subx.shape[0]
            slope = np.random.uniform(-self.slope_rate, self.slope_rate)
            x = np.arange(subseq_len)
            slope_component = slope * x
            sloped_subx = subx + slope_component
            sloped_subx_views.append(sloped_subx)

        if len(sloped_subx_views) == 1:
            sloped_subx_views = sloped_subx_views[0]
        return sloped_subx_views

    def add_noise_window(self, *subx_views):

        augmented_views = []

        for subx in subx_views:
            subseq_len = subx.shape[0]

            # Randomly select a window within the sequence
            window_size = int(subseq_len * self.window_ratio)
            window_start = np.random.randint(0, subseq_len - window_size + 1)
            window_end = window_start + window_size

            # Generate white noise for the window
            noise = np.random.normal(0, self.noise_std, size=(window_size,))

            # Apply white noise within the window
            augmented_view = subx.copy()
            augmented_view[window_start:window_end] += noise

            augmented_views.append(augmented_view)

        if len(augmented_views) == 1:
            augmented_views = augmented_views[0]

        return augmented_views

    def add_window_warp(self, *subx_views):
        # reference https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py

        warped_views = []
        window_ratio = self.window_ratio
        scales = [0.1, 1.1]  # Define the scales for the warp

        for subx in subx_views:
            n_timesteps = subx.shape[0]
            warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)
            window_start = np.random.randint(low=0, high=n_timesteps - warp_size)
            window_end = window_start + warp_size

            # Select a random scale for the warp
            scale = np.random.choice(scales)

            # Apply the warp to the window
            window_seg = np.interp(
                np.linspace(0, warp_size - 1, num=int(warp_size * scale)),
                np.arange(warp_size),
                subx[window_start:window_end],
            )

            # Concatenate the start segment, warped window, and end segment
            warped_subx = np.concatenate(
                (subx[:window_start], window_seg, subx[window_end:])
            )

            # Interpolate the warped sequence back to its original length
            warped_subx = np.interp(
                np.arange(n_timesteps),
                np.linspace(0, n_timesteps - 1, num=warped_subx.size),
                warped_subx,
            )

            warped_views.append(warped_subx)

        if len(warped_views) == 1:
            warped_views = warped_views[0]

        return warped_views

    def add_magnitude_warp(self, *subx_views, sigma=0.1, n_knots=4):
        # reference: https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py

        warped_views = []
        for subx in subx_views:
            n_timesteps = subx.shape[0]

            # Generate the original and warp steps
            orig_steps = np.arange(n_timesteps)
            warp_steps = np.linspace(0, n_timesteps - 1, num=n_knots + 2)

            # Generate a random warp for each feature
            random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_knots + 2,))

            # Apply the warp to each feature
            warper = scipy.interpolate.CubicSpline(warp_steps, random_warps)(orig_steps)
            warped_subx = subx * warper

            warped_views.append(warped_subx)

        if len(warped_views) == 1:
            warped_views = warped_views[0]

        return warped_views

    def add_slice_and_shuffle(self, *subx_views):
        """
        Slice the input sequences into segments and shuffle them.
        """
        shuffled_subx_views = []
        n_segments = self.n_segments

        for subx in subx_views:
            # Randomly pick n_segments-1 points where to slice
            idxs = np.random.randint(0, subx.shape[0], size=n_segments - 1)
            slices = []
            start_idx = 0
            for j in sorted(idxs):
                s = subx[start_idx:j]
                start_idx = j
                slices.append(s)
            slices.append(subx[start_idx:])
            np.random.shuffle(slices)
            shuffled_subx = np.concatenate(slices)
            shuffled_subx_views.append(shuffled_subx)

        if len(shuffled_subx_views) == 1:
            shuffled_subx_views = shuffled_subx_views[0]

        return shuffled_subx_views


class TimeFreqAugmenter(object):
    def __init__(
        self,
        n_fft=16,
        mask_density=0.2,
        rotation_max_angle=10.0,
        min_scale=0.8,
        max_scale=1.2,
        block_size_scale=0.1,
        block_density=0.2,
        gaus_mean=0,
        gaus_std=0.01,
        num_bands_to_remove=1,
        band_scale_factor=0.1,
        phase_max_change=np.pi / 4,
        max_shear_x=0.1,
        max_shear_y=0.1,
        **kwargs,
    ):
        self.n_fft = n_fft
        self.mask_density = mask_density
        self.rotation_max_angle = rotation_max_angle
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.block_size_scale = block_size_scale
        self.block_density = block_density
        self.gaus_mean = gaus_mean
        self.gaus_std = gaus_std
        self.num_bands_to_remove = num_bands_to_remove
        self.band_scale_factor = band_scale_factor
        self.phase_max_change = phase_max_change
        self.max_shear_x = max_shear_x
        self.max_shear_y = max_shear_y

        self.method_mapping = {
            "random_masks": "add_random_masks_augmentation",
            "rotation": "add_rotation_augmentation",
            "scale": "add_scale_augmentation",
            "block": "add_block_augmentation",
            "gaussian": "add_gaussian_augmentation",
            "band": "add_band_augmentation",
            "phase": "add_phase_augmentation",
            "shear": "add_shear_augmentation",
        }

    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def apply_augmentation(self, method_name, input):
        if method_name in self.method_mapping:
            method = getattr(self, self.method_mapping[method_name])
            return method(input)
        else:
            raise ValueError(
                f"{method_name} is not a valid method for time-frequency augmentation"
            )

    def apply_augmentations(self, method_names, input):
        for method_name in method_names:
            input = self.apply_augmentation(method_name, input)
        return input

    def stft(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.copy(), dtype=torch.float32)

        window = torch.hann_window(self.n_fft)
        return torch.stft(
            x, n_fft=self.n_fft, window=window, return_complex=True, onesided=False
        )

    def istft(self, u, original_length):
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u.copy(), dtype=torch.float32)

        if u.ndim == 1:
            u = u.unsqueeze(0)

        window = torch.hann_window(self.n_fft)
        istft_output = torch.istft(u, n_fft=self.n_fft, window=window, onesided=False)

        # Adjust the length of the output
        if istft_output.shape[-1] < original_length:
            pad_length = original_length - istft_output.shape[-1]
            istft_output = F.pad(istft_output, (0, pad_length))
        elif istft_output.shape[-1] > original_length:
            istft_output = istft_output[..., :original_length]

        return istft_output

    # --- Augmentation methods ---
    def add_random_masks_augmentation(self, *time_frequency_views):
        """
        Add random zero masks to time-frequency arrays.

        Parameters:
        - time_frequency_views: List of time-frequency arrays.

        Returns:
        - masked_views: List of time-frequency arrays with random zero masks.
        """
        if not (0 <= self.mask_density <= 1):
            raise ValueError("mask_density should be between 0 and 1")

        masked_views = []

        for time_frequency_view in time_frequency_views:
            masked_view = time_frequency_view.copy()

            # Determine the number of zeros to add based on the mask_density
            num_zeros = int(self.mask_density * time_frequency_view.size)

            # Generate random indices to set to zero
            mask_indices = np.random.choice(
                time_frequency_view.size, num_zeros, replace=False
            )

            # Set the selected indices to zero
            masked_view.ravel()[mask_indices] = 0

            masked_views.append(masked_view)

        if len(masked_views) == 1:
            masked_views = masked_views[0]

        return masked_views

    def add_rotation_augmentation(self, *time_frequency_arrays):
        """
        Add random rotations to time-frequency arrays.

        Parameters:
        - time_frequency_arrays: List of time-frequency arrays.

        Returns:
        - rotated_arrays: List of time-frequency arrays with random rotations.
        """
        if self.rotation_max_angle < 0:
            raise ValueError("max_angle should be a non-negative value")

        rotated_arrays = []

        for time_frequency_array in time_frequency_arrays:
            # Generate a random rotation angle between -max_angle and max_angle
            rotation_angle = np.random.uniform(
                -self.rotation_max_angle, self.rotation_max_angle
            )

            # Perform the rotation using scipy's rotate function
            rotated_array = rotate(
                time_frequency_array, angle=rotation_angle, reshape=False
            )
            rotated_arrays.append(rotated_array)

        if len(rotated_arrays) == 1:
            rotated_arrays = rotated_arrays[0]

        return rotated_arrays

    def add_shear_augmentation(self, *time_frequency_arrays):
        """
        Apply shear augmentation to time-frequency arrays in randomly chosen x or y direction.

        Parameters:
        - time_frequency_arrays: List of time-frequency arrays.

        Returns:
        - sheared_arrays: List of time-frequency arrays with shear applied.
        """
        sheared_arrays = []

        for time_frequency_array in time_frequency_arrays:
            # Randomly choose the direction (x or y)
            shear_direction = np.random.choice(["x", "y"])

            if shear_direction == "x":
                # Calculate the maximum shear value based on array width
                shear_x = np.random.uniform(-self.max_shear_x, self.max_shear_x)
                shear_y = 0.0
            else:  # Shear in y direction
                # Calculate the maximum shear value based on array height
                shear_x = 0.0
                shear_y = np.random.uniform(-self.max_shear_y, self.max_shear_y)

            # Apply shear transformation using affine_transform
            # sheared_array = affine_transform(time_frequency_array, matrix=[[1.0, shear_x], [shear_y, 1.0]], mode='constant')
            sheared_array = affine_transform(
                time_frequency_array,
                matrix=[[1.0, shear_x], [shear_y, 1.0]],
                mode="constant",
            )

            sheared_arrays.append(sheared_array)

        if len(sheared_arrays) == 1:
            sheared_arrays = sheared_arrays[0]

        return sheared_arrays

    def add_scale_augmentation(self, *time_frequency_views):
        """
        Add random amplitude scaling to time-frequency arrays.

        Parameters:
        - time_frequency_views: List of time-frequency arrays.

        Returns:
        - scaled_views: List of time-frequency arrays with random amplitude scaling.
        """
        if self.min_scale < 0 or self.max_scale < self.min_scale:
            raise ValueError("Invalid scaling factor values")

        scaled_views = []

        for time_frequency_view in time_frequency_views:
            # Generate a random scaling factor between min_scale and max_scale
            scaling_factor = np.random.uniform(self.min_scale, self.max_scale)

            # Apply the scaling factor to the magnitudes while keeping the phase information
            scaled_view = (
                np.abs(time_frequency_view)
                * scaling_factor
                * np.exp(1j * np.angle(time_frequency_view))
            )
            scaled_views.append(scaled_view)

        if len(scaled_views) == 1:
            scaled_views = scaled_views[0]

        return scaled_views

    def add_block_augmentation(self, *time_frequency_views):
        if self.block_density < 0 or self.block_density > 1:
            raise ValueError("Density should be between 0 and 1")

        if self.block_size_scale <= 0 or self.block_size_scale > 1:
            raise ValueError("Invalid scale factor")

        augmented_views = []

        for time_frequency_view in time_frequency_views:
            # Create a copy of the input array to apply augmentation
            augmented_view = time_frequency_view.copy()

            # Calculate the block size based on the scale factor and input dimensions
            block_height = int(augmented_view.shape[0] * self.block_size_scale)
            block_width = int(augmented_view.shape[1] * self.block_size_scale)

            # Determine the number of blocks to add
            num_blocks = int(
                self.block_density * (augmented_view.shape[0] * augmented_view.shape[1])
            )

            for _ in range(num_blocks):
                # Randomly choose a position for the block
                block_x = np.random.randint(
                    0, augmented_view.shape[0] - block_height + 1
                )
                block_y = np.random.randint(
                    0, augmented_view.shape[1] - block_width + 1
                )

                real_block = np.zeros((block_height, block_width), dtype=np.float32)
                imag_block = np.zeros((block_height, block_width), dtype=np.float32)

                # Add the block to the time-frequency representation
                augmented_view[
                    block_x : block_x + block_height, block_y : block_y + block_width
                ] = real_block
                augmented_view[
                    block_x : block_x + block_height, block_y : block_y + block_width
                ] += (1j * imag_block)

            # Pad the augmented array to match the input size if needed
            pad_height = time_frequency_view.shape[0] - augmented_view.shape[0]
            pad_width = time_frequency_view.shape[1] - augmented_view.shape[1]
            if pad_height > 0 or pad_width > 0:
                augmented_view = np.pad(
                    augmented_view, ((0, pad_height), (0, pad_width)), mode="constant"
                )

            augmented_views.append(augmented_view)

        if len(augmented_views) == 1:
            augmented_views = augmented_views[0]

        return augmented_views

    def add_gaussian_augmentation(self, *time_frequency_views):
        """
        Add Gaussian noise to time-frequency arrays.

        Parameters:
        - time_frequency_views: List of time-frequency arrays.

        Returns:
        - noisy_views: List of time-frequency arrays with added Gaussian noise.
        """
        if self.gaus_std <= 0:
            raise ValueError("std (standard deviation) should be a positive value")

        noisy_views = []

        for time_frequency_view in time_frequency_views:
            # Generate Gaussian noise with the specified mean and standard deviation
            noise = np.random.normal(
                self.gaus_mean, self.gaus_std, time_frequency_view.shape
            )

            # Add the generated noise to the original time-frequency array
            noisy_view = time_frequency_view + noise
            noisy_views.append(noisy_view)

        if len(noisy_views) == 1:
            noisy_views = noisy_views[0]

        return noisy_views

    def add_band_augmentation(self, *time_frequency_views):
        """
        Perform random scaled band augmentation on time-frequency arrays by removing random frequency bands.

        Parameters:
        - time_frequency_views: List of time-frequency arrays.

        Returns:
        - augmented_views: List of time-frequency arrays with random scaled bands removed.
        """
        if (
            self.num_bands_to_remove < 0
            or self.num_bands_to_remove >= time_frequency_views[0].shape[0]
        ):
            raise ValueError("Invalid number of bands to remove")

        if self.band_scale_factor <= 0 or self.band_scale_factor > 1:
            raise ValueError("Invalid scale factor")

        augmented_views = []

        for time_frequency_view in time_frequency_views:
            # Calculate the band width based on the scale factor and input representation height
            band_width = int(time_frequency_view.shape[0] * self.band_scale_factor)

            # Create a copy of the input array to apply augmentation
            augmented_view = time_frequency_view.copy()

            for _ in range(self.num_bands_to_remove):
                # Randomly choose a starting frequency for the band
                start_band = np.random.randint(
                    0, time_frequency_view.shape[0] - band_width + 1
                )

                # Remove the selected band(s)
                augmented_view[start_band : start_band + band_width, :] = 0

            augmented_views.append(augmented_view)

        if len(augmented_views) == 1:
            augmented_views = augmented_views[0]

        return augmented_views

    def add_phase_augmentation(self, *time_frequency_views):
        """
        Apply phase augmentation to time-frequency array views.

        Parameters:
        - time_frequency_views: List of time-frequency arrays.

        Returns:
        - augmented_views: List of time-frequency arrays with complex phase augmentation.
        """
        augmented_views = []

        for time_frequency_view in time_frequency_views:
            augmented_views = []

            n_channels, subseq_len = time_frequency_view.shape
            augmented_view = np.zeros((n_channels, subseq_len), dtype=np.complex64)

            for i in range(n_channels):
                # Modify the phase of the time-frequency view while controlling phase changes
                phase = np.angle(time_frequency_view[i])
                phase_change = np.random.uniform(
                    -self.phase_max_change, self.phase_max_change
                )
                augmented_phase = phase + phase_change

                # Reconstruct the augmented time-frequency view with modified phase
                augmented_view[i] = np.abs(time_frequency_view[i]) * np.exp(
                    1j * augmented_phase
                )

            augmented_views.append(augmented_view)

        if len(augmented_views) == 1:
            augmented_views = augmented_views[0]

        return augmented_views
