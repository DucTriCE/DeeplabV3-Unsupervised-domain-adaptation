# import torch
# import torch.nn.functional as F

# batch_size, num_classes, height, width = 4, 3, 5, 5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# outputs = torch.rand((batch_size, num_classes, height, width)).to(device).log_softmax(dim=1).exp()
# targets = torch.randint(0, 2, (batch_size, num_classes, height, width)).to(device)


# values, indices = torch.max(outputs, 1)  # Get max values and their indices
# labels_new = torch.where(values > 0.5, indices, torch.full_like(indices, num_classes))

# targets_one_hot = F.one_hot(labels_new, num_classes + 1).float()
# targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # Adjust dimensions to [batch, channel, height, width]
# targets_final = targets_one_hot[:, :num_classes, :, :]

# mask = targets_final != 0       #([4,3,5,5])
# # outputs_one_hot = F.one_hot(labels_new, num_classes + 1).float()
# # outputs_one_hot = outputs_one_hot.permute(0, 3, 1, 2)  # Adjust dimensions to [batch, channel, height, width]
# # outputs_final = outputs_one_hot[:, :num_classes, :, :]
# outputs_final = outputs*mask

# print(f'First target shape: {targets.shape}')
# print(targets)
# print(f'Final target shape: {targets_final.shape}')
# print(targets_final)

# print(f'First output shape: {outputs.shape}')
# print(outputs)
# print(f'Final output shape: {outputs_final.shape}')
# print(outputs_final)

import torch
import torch.nn.functional as F

batch_size, num_classes, height, width = 4, 3, 5, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# y_pred ([4, 3, 500, 500])     y_true([4, 500, 500])


y_pred = torch.rand((batch_size, num_classes, height, width)).log_softmax(dim=1).exp()
y_true = torch.randint(0, 3, (batch_size, height, width))

y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
y_true = y_true.permute(0, 3, 1, 2)  # N, C, H*W
print(y_pred, "TRUE", y_true)

values, indices = torch.max(y_pred, 1)  # Get max values and their indices
labels_new = torch.where(values > 0.5, indices, torch.full_like(indices, num_classes))

y_true_one_hot = F.one_hot(labels_new, num_classes + 1)
y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)  
y_true_one_hot = y_true_one_hot[:, :num_classes, :]


y_pred = y_pred*y_true_one_hot
y_true = y_true*y_true_one_hot

print(y_pred.shape, y_true.shape)
