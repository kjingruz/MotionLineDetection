from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the model
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for data in dataloader:
        inputs = data
        labels = torch.ones(inputs.size(0), 92)  # Placeholder labels
        outputs = model(inputs)
        preds = (outputs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1 Score: {f1:.3f}')
