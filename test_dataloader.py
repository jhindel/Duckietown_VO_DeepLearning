from dataloader import DuckietownDataset

test_loader = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images", transform=None, target_transform=None)

for test_images, test_labels in test_loader:
    sample_image = test_images[0]
    sample_label = test_labels[0]

"""
mean = 0.
std = 0.
for images, _ in test_loader:
    mean += images.mean(axis=(0, 1))
    std += images.std(axis=(0, 1))

mean /= len(test_loader)
std /= len(test_loader)

print(mean)
print(std)
"""