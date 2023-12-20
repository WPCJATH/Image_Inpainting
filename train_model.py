

import subprocess
def inference_deep_method(image_path, mask_path, out_path):
    cmds = ["call activate reprod && python", "DeepMethodBaseline/test.py", 
            "--image", image_path, "--mask", mask_path, "--out", out_path,
            "--checkpoint model/pretrained/states_pt_places2.pth"]
    process = subprocess.call(" ".join(cmds), shell=True, stdout=subprocess.DEVNULL)

    return process == 0


if __name__ == "__main__":
    image_path = "DeepMethodBaseline/examples/inpaint/case2.png"
    mask_path = "DeepMethodBaseline/examples/inpaint/case2_mask.png"
    out_path = "data/test2.png"
    print(inference_deep_method(image_path, mask_path, out_path))

