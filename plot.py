import re
import numpy as np

# Read the contents of the file
with open('outTwinL.out', 'r') as file:
    text = file.read()

# Pattern to extract values before specific keywords
pattern = r'(?<=Acc\()([\d.]+)|(?<=IOU \()\s*([\d.]+)|(?<=mIOU\()\s*([\d.]+)|Learning rate: ([\d.e+-]+)|---\s(\d+)'

# Find all matches in the text
matches = re.findall(pattern, text)

# Group the matches by their respective categories
categories = ['Acc', 'IOU', 'mIOU', 'Learning rate', 'Epochs']
data = {category: [] for category in categories}

for index, match in enumerate(matches):
    for i in range(len(categories)):
        if match[i] != '':
            if categories[i] == 'Epochs':
                data[categories[i]].append(int(match[i]))  # Convert Epochs to int
            else:
                val = match[i]
                if 'e' in val:  # Check if it's scientific notation
                    data[categories[i]].append("{:.8f}".format(float(val)))
                else:
                    val = float(val)
                    data[categories[i]].append(val)

drive_categories = {}
for i, category in enumerate(categories):
    values = data[category]
    drive_values = values[::2]  # Get values at even indices
    if i == 0:
        drive_categories['Drive_' + category.lower()] = drive_values
    else:
        drive_categories['Drive_' + category.lower()] = drive_values

# Storing in different lists
drive_miou = drive_categories['Drive_miou']
# Display the extracted values
print("Drive mIOU:", len(drive_miou))


ema_span = 20
df_smoothed = df.ewm(span=ema_span).mean()
df_smoothed1 = df1.ewm(span=ema_span).mean()


# Choose specific epochs or summary statistics for comparison
epochs_to_compare = [50]  # Example: comparing at these specific epochs

# Selecting values at chosen epochs for TwinL and TwinM
drive_miou_at_epochs = df_smoothed.loc[epochs_to_compare]['Drive_mIOU']
drive_miou_at_epochs1 = df_smoothed1.loc[epochs_to_compare]['Drive_mIOU1']

lane_iou_at_epochs = df_smoothed.loc[epochs_to_compare]['Lane_IOU']
lane_iou_at_epochs1 = df_smoothed1.loc[epochs_to_compare]['Lane_IOU1']

# Creating bar plots for comparison
fig, axs = plt.subplots(2, figsize=(10, 8))

axs[0].bar(np.arange(len(epochs_to_compare))-0.2, drive_miou_at_epochs, width=0.4, label='TwinL', color='red')
axs[0].bar(np.arange(len(epochs_to_compare))+0.2, drive_miou_at_epochs1, width=0.4, label='TwinM', color='green')
axs[0].set_title('Drive mIOU at Specific Epochs')
axs[0].set_xticks(np.arange(len(epochs_to_compare)))
axs[0].set_xticklabels(epochs_to_compare)
axs[0].legend()

axs[1].bar(np.arange(len(epochs_to_compare))-0.2, lane_iou_at_epochs, width=0.4, label='TwinL', color='red')
axs[1].bar(np.arange(len(epochs_to_compare))+0.2, lane_iou_at_epochs1, width=0.4, label='TwinM', color='green')
axs[1].set_title('Lane IOU at Specific Epochs')
axs[1].set_xticks(np.arange(len(epochs_to_compare)))
axs[1].set_xticklabels(epochs_to_compare)
axs[1].legend()

plt.tight_layout()
plt.savefig('Bar_Plots.png')
plt.show()