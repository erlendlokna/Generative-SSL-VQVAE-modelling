class TimeFreqAugmentation(object):
    def __init__(self, block_size, mask_density=0.2, rotation_max_angle=10.0,
                 min_scale=0.8, max_scale=1.2, block_size_scale=0.1,
                 block_density=0.2, gaus_mean=0, gaus_std=0.01,
                 num_bands_to_remove=1, band_scale_factor=0.1,
                 phase_max_change=np.pi/4):
        
        self.block_size = block_size
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
        


    def add_random_masks(time_frequency_view, mask_density=0.2, **kwargs):
        """
        Add random zero masks to a time-frequency array.

        Parameters:
        - time_frequency_array (numpy.ndarray): The input time-frequency array.
        - mask_density (float): The density of the random zero masks. 
                                A value between 0 and 1, indicating the proportion of zeros to add.

        Returns:
        - masked_array (numpy.ndarray): The time-frequency array with random zero masks.
        """
        if not (0 <= mask_density <= 1):
            raise ValueError("mask_density should be between 0 and 1")

        masked_view = time_frequency_view.copy()

        # Determine the number of zeros to add based on the mask_density
        num_zeros = int(mask_density * time_frequency_view.size)

        # Generate random indices to set to zero
        mask_indices = np.random.choice(time_frequency_view.size, num_zeros, replace=False)

        # Set the selected indices to zero
        masked_view.ravel()[mask_indices] = 0

        return masked_view


    def add_rotation_augmentation(time_frequency_array, rotation_max_angle, **kwargs):
        """
        Add random rotations to a time-frequency array.

        Parameters:
        - time_frequency_array (numpy.ndarray): The input time-frequency array.
        - max_angle (float): The maximum rotation angle in degrees.

        Returns:
        - rotated_array (numpy.ndarray): The time-frequency array with random rotations.
        """
        if rotation_max_angle < 0:
            raise ValueError("max_angle should be a non-negative value")

        # Generate a random rotation angle between -max_angle and max_angle
        rotation_angle = np.random.uniform(-rotation_max_angle, rotation_max_angle)

        # Perform the rotation using scipy's rotate function
        rotated_array = rotate(time_frequency_array, angle=rotation_angle, reshape=False)

        return rotated_array


    def add_scale_augmentation(time_frequency_view, min_scale=0.8, max_scale=1.2, **kwargs):
        """
        Add random amplitude scaling to a time-frequency array.

        Parameters:
        - time_frequency_array (numpy.ndarray): The input time-frequency array.
        - min_scale (float): The minimum scaling factor to apply.
        - max_scale (float): The maximum scaling factor to apply.

        Returns:
        - scaled_array (numpy.ndarray): The time-frequency array with random amplitude scaling.
        """
        if min_scale < 0 or max_scale < min_scale:
            raise ValueError("Invalid scaling factor values")

        # Generate a random scaling factor between min_scale and max_scale
        scaling_factor = np.random.uniform(min_scale, max_scale)

        # Apply the scaling factor to the magnitudes while keeping the phase information
        scaled_view = np.abs(time_frequency_view) * scaling_factor * np.exp(1j * np.angle(time_frequency_view))

        return scaled_view

    def add_block_augmentation(time_frequency_view, block_size_scale=0.1, block_density=0.2, **kwargs):
        if block_density < 0 or block_density > 1:
            raise ValueError("Density should be between 0 and 1")

        if block_size_scale <= 0 or block_size_scale > 1:
            raise ValueError("Invalid scale factor")

        # Create a copy of the input array to apply augmentation
        augmented_view = time_frequency_view.copy()

        # Calculate the block size based on the scale factor and input dimensions
        block_height = int(augmented_view.shape[0] * block_size_scale)
        block_width = int(augmented_view.shape[1] * block_size_scale)

        # Determine the number of blocks to add
        num_blocks = int(block_density * (augmented_view.shape[0] * augmented_view.shape[1]))

        for _ in range(num_blocks):
            # Randomly choose a position for the block
            block_x = np.random.randint(0, augmented_view.shape[0] - block_height + 1)
            block_y = np.random.randint(0, augmented_view.shape[1] - block_width + 1)

            real_block = np.zeros((block_height, block_width), dtype=np.float32)
            imag_block = np.zeros((block_height, block_width), dtype=np.float32)

            # Add the block to the time-frequency representation
            augmented_view[block_x:block_x + block_height, block_y:block_y + block_width] = real_block
            augmented_view[block_x:block_x + block_height, block_y:block_y + block_width] += 1j * imag_block

        # Pad the augmented array to match the input size if needed
        pad_height = time_frequency_view.shape[0] - augmented_view.shape[0]
        pad_width = time_frequency_view.shape[1] - augmented_view.shape[1]
        if pad_height > 0 or pad_width > 0:
            augmented_view = np.pad(augmented_view, ((0, pad_height), (0, pad_width)), mode='constant')

        return augmented_view

    def add_gaussian_augmentation(time_frequency_view, gaus_mean=0, gaus_std=0.01, **kwargs):
        """
        Add Gaussian noise to a time-frequency array.

        Parameters:
        - time_frequency_array (numpy.ndarray): The input time-frequency array.
        - mean (float): The mean of the Gaussian noise distribution.
        - std (float): The standard deviation of the Gaussian noise distribution.

        Returns:
        - noisy_array (numpy.ndarray): The time-frequency array with added Gaussian noise.
        """
        if gaus_std <= 0:
            raise ValueError("std (standard deviation) should be a positive value")

        # Generate Gaussian noise with the specified mean and standard deviation
        noise = np.random.normal(gaus_mean, gaus_std, time_frequency_view.shape)

        # Add the generated noise to the original time-frequency array
        noisy_view = time_frequency_view + noise

        return noisy_view

    def add_band_augmentation(time_frequency_view, num_bands_to_remove=1, band_scale_factor=0.1, **kwargs):
        """
        Perform random scaled band augmentation on a time-frequency array by removing random frequency bands.

        Parameters:
        - time_frequency_array (numpy.ndarray): The input time-frequency array.
        - num_bands_to_remove (int): The number of frequency bands to randomly remove.
        - scale_factor (float): The scale factor for band width relative to the input representation.

        Returns:
        - augmented_array (numpy.ndarray): The time-frequency array with random scaled bands removed.
        """
        if num_bands_to_remove < 0 or num_bands_to_remove >= time_frequency_view.shape[0]:
            raise ValueError("Invalid number of bands to remove")

        if band_scale_factor <= 0 or band_scale_factor > 1:
            raise ValueError("Invalid scale factor")

        # Calculate the band width based on the scale factor and input representation height
        band_width = int(time_frequency_view.shape[0] * band_scale_factor)

        # Create a copy of the input array to apply augmentation
        augmented_view = time_frequency_view.copy()

        for _ in range(num_bands_to_remove):
            # Randomly choose a starting frequency for the band
            start_band = np.random.randint(0, time_frequency_view.shape[0] - band_width + 1)

            # Remove the selected band(s)
            augmented_view[start_band:start_band + band_width, :] = 0

        return augmented_view

    def add_phase_augmentation(time_frequency_view, phase_max_change=np.pi/4, **kwargs):
        """
        Apply phase augmentation to a single time-frequency array view.

        Parameters:
        - time_frequency_view (numpy.ndarray): The input time-frequency array view.

        Returns:
        - augmented_view (numpy.ndarray): The time-frequency array view with complex phase augmentation.
        """
        n_channels, subseq_len = time_frequency_view.shape
        augmented_view = np.zeros((n_channels, subseq_len), dtype=np.complex64)

        for i in range(n_channels):
            # Modify the phase of the time-frequency view while controlling phase changes
            phase = np.angle(time_frequency_view[i])
            phase_change = np.random.uniform(-phase_max_change, phase_max_change)
            augmented_phase = phase + phase_change

            # Reconstruct the augmented time-frequency view with modified phase
            augmented_view[i] = np.abs(time_frequency_view[i]) * np.exp(1j * augmented_phase)

        return augmented_view

def stft(x, n_fft, **kwargs):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.stft(x, n_fft=n_fft, return_complex=True, onesided=False)

def istft(u, n_fft, original_length, **kwargs):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    
    istft_output = torch.istft(u, n_fft=n_fft, return_complex=False, onesided=False)
    
    # Trim or zero-pad the ISTFT output to match the original length
    if len(istft_output) < original_length:
        pad_length = original_length - len(istft_output)
        istft_output = torch.cat((istft_output, torch.zeros(pad_length)))
    elif len(istft_output) > original_length:
        istft_output = istft_output[:original_length]
    
    return istft_output

def add_random_masks(time_frequency_array, mask_density=0.2, **kwargs):
    """
    Add random zero masks to a time-frequency array.

    Parameters:
    - time_frequency_array (numpy.ndarray): The input time-frequency array.
    - mask_density (float): The density of the random zero masks. 
                            A value between 0 and 1, indicating the proportion of zeros to add.

    Returns:
    - masked_array (numpy.ndarray): The time-frequency array with random zero masks.
    """
    if not (0 <= mask_density <= 1):
        raise ValueError("mask_density should be between 0 and 1")

    masked_array = time_frequency_array.copy()

    # Determine the number of zeros to add based on the mask_density
    num_zeros = int(mask_density * time_frequency_array.size)

    # Generate random indices to set to zero
    mask_indices = np.random.choice(time_frequency_array.size, num_zeros, replace=False)

    # Set the selected indices to zero
    masked_array.ravel()[mask_indices] = 0

    return masked_array


def add_rotation_augmentation(time_frequency_array, rotation_max_angle, **kwargs):
    """
    Add random rotations to a time-frequency array.

    Parameters:
    - time_frequency_array (numpy.ndarray): The input time-frequency array.
    - max_angle (float): The maximum rotation angle in degrees.

    Returns:
    - rotated_array (numpy.ndarray): The time-frequency array with random rotations.
    """
    if rotation_max_angle < 0:
        raise ValueError("max_angle should be a non-negative value")

    # Generate a random rotation angle between -max_angle and max_angle
    rotation_angle = np.random.uniform(-rotation_max_angle, rotation_max_angle)

    # Perform the rotation using scipy's rotate function
    rotated_array = rotate(time_frequency_array, angle=rotation_angle, reshape=False)

    return rotated_array


def add_scale_augmentation(time_frequency_array, min_scale=0.8, max_scale=1.2, **kwargs):
    """
    Add random amplitude scaling to a time-frequency array.

    Parameters:
    - time_frequency_array (numpy.ndarray): The input time-frequency array.
    - min_scale (float): The minimum scaling factor to apply.
    - max_scale (float): The maximum scaling factor to apply.

    Returns:
    - scaled_array (numpy.ndarray): The time-frequency array with random amplitude scaling.
    """
    if min_scale < 0 or max_scale < min_scale:
        raise ValueError("Invalid scaling factor values")

    # Generate a random scaling factor between min_scale and max_scale
    scaling_factor = np.random.uniform(min_scale, max_scale)

    # Apply the scaling factor to the magnitudes while keeping the phase information
    scaled_array = np.abs(time_frequency_array) * scaling_factor * np.exp(1j * np.angle(time_frequency_array))

    return scaled_array

def add_block_augmentation(u, block_size_scale=0.1, block_density=0.2, **kwargs):
    if block_density < 0 or block_density > 1:
        raise ValueError("Density should be between 0 and 1")

    if block_size_scale <= 0 or block_size_scale > 1:
        raise ValueError("Invalid scale factor")

    # Create a copy of the input array to apply augmentation
    augmented_array = u.copy()

    # Calculate the block size based on the scale factor and input dimensions
    block_height = int(augmented_array.shape[0] * block_size_scale)
    block_width = int(augmented_array.shape[1] * block_size_scale)

    # Determine the number of blocks to add
    num_blocks = int(block_density * (augmented_array.shape[0] * augmented_array.shape[1]))

    for _ in range(num_blocks):
        # Randomly choose a position for the block
        block_x = np.random.randint(0, augmented_array.shape[0] - block_height + 1)
        block_y = np.random.randint(0, augmented_array.shape[1] - block_width + 1)

        real_block = np.zeros((block_height, block_width), dtype=np.float32)
        imag_block = np.zeros((block_height, block_width), dtype=np.float32)

        # Add the block to the time-frequency representation
        augmented_array[block_x:block_x + block_height, block_y:block_y + block_width] = real_block
        augmented_array[block_x:block_x + block_height, block_y:block_y + block_width] += 1j * imag_block

    # Pad the augmented array to match the input size if needed
    pad_height = u.shape[0] - augmented_array.shape[0]
    pad_width = u.shape[1] - augmented_array.shape[1]
    if pad_height > 0 or pad_width > 0:
        augmented_array = np.pad(augmented_array, ((0, pad_height), (0, pad_width)), mode='constant')

    return augmented_array

def add_gaussian_augmentation(time_frequency_array, gaus_mean=0, gaus_std=0.01, **kwargs):
    """
    Add Gaussian noise to a time-frequency array.

    Parameters:
    - time_frequency_array (numpy.ndarray): The input time-frequency array.
    - mean (float): The mean of the Gaussian noise distribution.
    - std (float): The standard deviation of the Gaussian noise distribution.

    Returns:
    - noisy_array (numpy.ndarray): The time-frequency array with added Gaussian noise.
    """
    if gaus_std <= 0:
        raise ValueError("std (standard deviation) should be a positive value")

    # Generate Gaussian noise with the specified mean and standard deviation
    noise = np.random.normal(gaus_mean, gaus_std, time_frequency_array.shape)

    # Add the generated noise to the original time-frequency array
    noisy_array = time_frequency_array + noise

    return noisy_array

def add_band_augmentation(time_frequency_array, num_bands_to_remove=1, band_scale_factor=0.1, **kwargs):
    """
    Perform random scaled band augmentation on a time-frequency array by removing random frequency bands.

    Parameters:
    - time_frequency_array (numpy.ndarray): The input time-frequency array.
    - num_bands_to_remove (int): The number of frequency bands to randomly remove.
    - scale_factor (float): The scale factor for band width relative to the input representation.

    Returns:
    - augmented_array (numpy.ndarray): The time-frequency array with random scaled bands removed.
    """
    if num_bands_to_remove < 0 or num_bands_to_remove >= time_frequency_array.shape[0]:
        raise ValueError("Invalid number of bands to remove")

    if band_scale_factor <= 0 or band_scale_factor > 1:
        raise ValueError("Invalid scale factor")

    # Calculate the band width based on the scale factor and input representation height
    band_width = int(time_frequency_array.shape[0] * band_scale_factor)

    # Create a copy of the input array to apply augmentation
    augmented_view = time_frequency_array.copy()

    for _ in range(num_bands_to_remove):
        # Randomly choose a starting frequency for the band
        start_band = np.random.randint(0, time_frequency_array.shape[0] - band_width + 1)

        # Remove the selected band(s)
        augmented_view[start_band:start_band + band_width, :] = 0

    return augmented_view

def add_phase_augmentation(time_frequency_view, phase_max_change=np.pi/4, **kwargs):
    """
    Apply phase augmentation to a single time-frequency array view.

    Parameters:
    - time_frequency_view (numpy.ndarray): The input time-frequency array view.

    Returns:
    - augmented_view (numpy.ndarray): The time-frequency array view with complex phase augmentation.
    """
    n_channels, subseq_len = time_frequency_view.shape
    augmented_view = np.zeros((n_channels, subseq_len), dtype=np.complex64)

    for i in range(n_channels):
        # Modify the phase of the time-frequency view while controlling phase changes
        phase = np.angle(time_frequency_view[i])
        phase_change = np.random.uniform(-phase_max_change, phase_max_change)
        augmented_phase = phase + phase_change

        # Reconstruct the augmented time-frequency view with modified phase
        augmented_view[i] = np.abs(time_frequency_view[i]) * np.exp(1j * augmented_phase)

    return augmented_view


def apply_timefreq_augmentations(x, aug_params):
    """
    Apply a random combination of augmentations to the input data.

    Parameters:
    - input_data: The input data to be augmented.
    - num_augmentations: The number of augmentations to apply (1 to 4).

    Returns:
    - augmented_data: The augmented data.
    - augmentation_combination: A list of the augmentation functions applied.
    """
    num_augmentations = np.random.choice(aug_params['num_augmentations']) + 1
    n_fft = aug_params['n_fft']

    augmentations = [
        ('Rotation', add_rotation_augmentation),
        ('Band', add_band_augmentation),
        ('Block', add_block_augmentation),
        ('phase', add_phase_augmentation),
        ('random_masking', add_random_masks),
        ('scaling', add_scale_augmentation),
        #('jitter', add_gaussian_augmentation)
    ]

    # Shuffle the list of augmentations and select the first 'num_augmentations'
    np.random.shuffle(augmentations)
    selected_augmentations = augmentations[:num_augmentations]

    # Apply selected augmentations
    u = stft(x, n_fft)
    augmented_data = stft(x, n_fft)
    augmentation_combination = []
    
    augmented_data = augmented_data.numpy()

    for augmentation_name, augmentation_function in selected_augmentations:
        augmented_data = augmentation_function(augmented_data, **aug_params)
        augmentation_combination.append(augmentation_name)

    augmented_data = torch.from_numpy(augmented_data)

    xaug = istft(augmented_data, n_fft, len(x))
    return xaug, augmentation_combination