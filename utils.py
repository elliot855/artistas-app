def generar_gradcam(model, imagen_tensor):
    gradientes = []
    activaciones = []

    def backward_hook(module, grad_in, grad_out):
        gradientes.append(grad_out[0])

    def forward_hook(module, input, output):
        activaciones.append(output)

    # 👇 CAMBIO AQUÍ
    capa_objetivo = model.model.conv_head

    capa_objetivo.register_forward_hook(forward_hook)
    capa_objetivo.register_backward_hook(backward_hook)

    output = model(imagen_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grads = gradientes[0]
    acts = activaciones[0]

    pesos = grads.mean(dim=(2, 3), keepdim=True)
    cam = (pesos * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (224, 224))

    return cam
