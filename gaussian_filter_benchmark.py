import cv2
import numpy as np
import time
import csv


# Function to generate a synthetic image of given size
def generate_image(size):
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)


# Function to measure execution time of a given operation
def measure_time(func, iterations=10):
    total_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        func()
        total_time += time.perf_counter() - start_time
    return total_time / iterations  # Return average time


# Function to apply 2D Gaussian filter
def apply_2d_gaussian(image, kernel_size, sigma):
    gaussian_kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.T
    return cv2.filter2D(image, ddepth=-1, kernel=gaussian_kernel_2d)


# Function to apply two 1D Gaussian filters (optimized)
def apply_1d_gaussian(image, kernel_size, sigma):
    gaussian_kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    return cv2.sepFilter2D(
        image, ddepth=-1, kernelX=gaussian_kernel_1d, kernelY=gaussian_kernel_1d
    )


# Function to save results into a CSV file
def save_results_to_csv(filename, results, headers):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)


# Main script
def main():
    # Parameters
    image_sizes = [
        512,
        1024,
        2048,
        4096,
        8192,
    ]  #  Varying image sizes for image size tests
    kernel_sizes = [
        3,
        5,
        9,
        15,
        21,
        31,
        41,
    ]  # Varying kernel sizes for kernel size tests
    sigma = 5  # Standard deviation for Gaussian kernel
    iterations = 20  # Number of iterations for averaging

    # Results for image size variation
    results_image_sizes = []

    # Test with varying image sizes
    for size in image_sizes:
        print(f"Testing with image size: {size}x{size}")

        # Generate synthetic image
        image = generate_image(size)

        # Measure time for 2D Gaussian
        time_2d = measure_time(lambda: apply_2d_gaussian(image, 15, sigma), iterations)

        # Measure time for 1D Gaussian applied twice
        time_1d = measure_time(lambda: apply_1d_gaussian(image, 15, sigma), iterations)

        # Save results
        results_image_sizes.append((size, time_2d, time_1d))
        print(
            f"Image Size: {size}x{size}, 2D Time: {time_2d:.6f}s, 1D x2 Time: {time_1d:.6f}s"
        )

    # Save results to a CSV file
    save_results_to_csv(
        "gaussian_filter_image_sizes.csv",
        results_image_sizes,
        ["Image Size", "2D Gaussian Time (s)", "1D Gaussian x2 Time (s)"],
    )
    print("Results saved in gaussian_filter_image_sizes.csv")

    # Results for kernel size variation
    results_kernel_sizes = []

    # Use a large fixed image for kernel size variation
    large_image = generate_image(4096)
    print(f"\nTesting with fixed image size: 4096x4096 for varying kernel sizes")

    # Test with varying kernel sizes
    for kernel_size in kernel_sizes:
        print(f"Testing with kernel size: {kernel_size}x{kernel_size}")

        # Measure time for 2D Gaussian
        time_2d = measure_time(
            lambda: apply_2d_gaussian(large_image, kernel_size, sigma), iterations
        )

        # Measure time for 1D Gaussian applied twice
        time_1d = measure_time(
            lambda: apply_1d_gaussian(large_image, kernel_size, sigma), iterations
        )

        # Save results
        results_kernel_sizes.append((kernel_size, time_2d, time_1d))
        print(
            f"Kernel Size: {kernel_size}x{kernel_size}, 2D Time: {time_2d:.6f}s, 1D x2 Time: {time_1d:.6f}s"
        )

    # Save results to a CSV file
    save_results_to_csv(
        "gaussian_filter_kernel_sizes.csv",
        results_kernel_sizes,
        ["Kernel Size", "2D Gaussian Time (s)", "1D Gaussian x2 Time (s)"],
    )
    print("Results saved in gaussian_filter_kernel_sizes.csv")


if __name__ == "__main__":
    main()
