import os
os.environ["PYRLTOOLS_IGNORE_TORCH_WARNING"] = "1"

if not "PYRLTOOLS_DISABLE_PYTORCH_IMPORT_TEST" in os.environ:
    import torch
    import torch.nn as nn
    from pyrltools import CACHE_PATH
    from pyrltools.onnx import load_mlp, evaluate
    from pyrltools.onnx import render
    from pyrltools import load_checkpoint_from_path

def test_pytorch_import():
    if "PYRLTOOLS_DISABLE_PYTORCH_IMPORT_TEST" in os.environ:
        return
    torch.manual_seed(0)

    n_input_features = 4
    n_hidden = 16
    n_output_features = 10

    model = nn.Sequential(
        nn.Linear(n_input_features, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_output_features)
    )

    model.eval()
    dummy_input = torch.randn(1, n_input_features)
    test_path = os.path.join(CACHE_PATH, "tests", "onnx")
    os.makedirs(test_path, exist_ok=True)
    model_path = os.path.join(test_path, "two_layer_net.onnx")
    torch.onnx.export(model, dummy_input, model_path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    sequential = load_mlp(model_path)
    loaded_output = evaluate(sequential, dummy_input.numpy())
    assert torch.allclose(model(dummy_input), torch.tensor(loaded_output), atol=1e-6)


    output_path = os.path.join(test_path, "two_layer_net.h")
    rendered = render(sequential)
    with open(output_path, "w") as f:
        f.write(rendered)



    rlt_model = load_checkpoint_from_path(output_path, interface_name="test_pytorch_import")
    loaded_output2 = rlt_model.evaluate(dummy_input.numpy()[0])
    max_output_diff = (torch.tensor(loaded_output2) - model(dummy_input)).abs().max()
    print("Max output diff: ", max_output_diff)
    assert torch.allclose(model(dummy_input), torch.tensor(loaded_output2), atol=1e-6)



if __name__ == "__main__":
    test_pytorch_import()