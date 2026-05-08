import torch
X = torch.tensor([[1],[2],[3]], requires_grad=False)
y = torch.tensor([[1],[2],[3]], requires_grad=False)

print(X)
print(y)
print("shape: ",X.shape)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w)
print(b)

def forward(X):
    return w*X + b
lr=0.5
for epoch in range(1000):
    y_pred = forward(X)
    loss= ((y-y_pred)**2).mean()
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
    w.grad.zero_()
    b.grad.zero_()
    if epoch%100==0:
        print("Epoch: ",epoch, "Loss: ", loss.item())

print("w: ",w)
print("b: ",b)
