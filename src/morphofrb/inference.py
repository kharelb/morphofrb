import sys
import time
import json
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from importlib.resources import files


from .model import CustomConvnext
from .weights import get_weights_path
from .inference_dataloader import load_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    

def typewriter(text, delay=0.001):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print() # Add a newline at the end

@torch.inference_mode()
def predict(model_state_dict_path=None,
            files_path=Path.cwd(),
            save_prediction_json=False,
            out_dir=None,
            batch_size=32,
            device=get_device()):
    predictions = []
    filenames = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir:
        save_file_name = f"{str(Path(out_dir))}/prediction_{timestamp}"
    else:
        save_file_name = f"prediction_{timestamp}"

    if model_state_dict_path is None:
        model_state_dict_path = get_weights_path()
    model = CustomConvnext(weight=None)
    model.load_state_dict(torch.load(model_state_dict_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loading model to: {device} for inference...")
    model.to(device)
    

    if Path(files_path).is_dir():
        dataloader = load_data(files_path, batch_size=batch_size)
        for img, names in dataloader:
            filenames.extend(list(names))
            logits = model(img.to(device))
            pred_prob = torch.sigmoid(logits)
            predictions.extend(pred_prob.squeeze().tolist())

    if Path(files_path).is_file():
        img, name = load_data(files_path)
        filenames.append(name)
        logits = model(img.to(device))
        pred_prob = torch.sigmoid(logits)
        predictions.append(pred_prob.squeeze().item())

    if save_prediction_json:
        with open(f'{save_file_name}.json', 'w') as f:
            print(f"Saving file as: {save_file_name}.json...")
            json.dump({'filenames': filenames,
                      'predictions': predictions},
                      f)
        
    return predictions, filenames


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--state_dict", type=str, help='state_dict of fine tuned model (by default fine tuned on CHIME Catalog 2 is loaded)')
    parser.add_argument("--files_path", type=str, required=True, help="Directory containing '.npy' files or path to '.npy' file.")
    parser.add_argument("--save_prediction", action="store_true", help="Save the prediction for files in a json?")
    parser.add_argument("--out_dir", type=str, help="Path of directory to save the prediction json file")
    parser.add_argument("--batch_size", type=int, help="Batch size for prediction. Depends on the system memory you use.")
    parser.add_argument("--print_results", action="store_true", help="Print the results on the CLI screen")

    args = parser.parse_args(argv)

    state_dict_path = args.state_dict
    files_path = args.files_path if args.files_path else Path.cwd()
    save_prediction = True if args.save_prediction else False
    out_dir = args.out_dir if args.out_dir else Path.cwd()
    batch_size = args.batch_size if args.batch_size else 32
    
    predictions, filenames = predict(model_state_dict_path=state_dict_path,
            files_path=files_path,
            save_prediction_json=save_prediction, 
            out_dir=out_dir, 
            batch_size=batch_size)

    if args.print_results:
        print("\n")
        typewriter("-"*44+" "*4+"-"*15)
        typewriter(" "*19+"Filename"+" "*19+" "*4+"Probability")
        typewriter("-"*44+" "*4+"-"*15)
        for i, j in zip(predictions, filenames):
            typewriter(f"- {j} -----> {round(i, 3)}")


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    
    
    
    