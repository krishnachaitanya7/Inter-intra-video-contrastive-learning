import torch


if __name__ == "__main__":
    a = torch.randn(4, 4)
    print(a)
    print(torch.argmax(a))
