import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def decode(model, src):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src]).to(device)
        pad_idx = hs.PL_voc[dataset.PAD]
        parent0 = [hs.PL_voc['root']]
        name0 = [hs.PL_voc['root']]
        trg0 = [hs.PL_voc[dataset.SOS]]
        output = model(src, torch.tensor([parent0]).to(device), torch.tensor([name0]).to(device),
                       torch.tensor([trg0]).to(device))
        print(output.shape)
        maxi = output[0][0].max(-1)[1].item()
        print(hs.PL_dict[maxi])


if __name__ == "__main__":
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    model = torch.load('models/transformer-seq2seq.pt')
    X_test, Y_test = hs.dataset('test')
    for i in range(len(X_test)):
        X = X_test[i]
        print(X)
        decode(model, X)
        input()