q_model.load_state_dict(torch.load(q_model_path))
model.eval()
model.to('cpu')
q_model.eval()
q_model.to('cpu')

inputs, labels = next(dataiter)
inputs.to('cpu')
labels.to('cpu')
true_labels = np.array(labels)

def run_test(net, images):
    with torch.no_grad():
        out = net(images)
        _, preds = torch.max(out, 1)
        predict_labels = np.array(preds)

    accuracy = accuracy_score(true_labels, predict_labels)
    return accuracy

acc = run_test(model, inputs)
print("Standard: ", acc)
acc = run_test(q_model, inputs)
print("Quantise: ", acc)