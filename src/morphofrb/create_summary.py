from torchinfo import summary

def create_summary(model,
                   input_size=(32, 3, 224, 224),
                   col_names=["input_size", "output_size", "num_params", "trainable"],
                   col_width=18,
                   row_settings=["var_names"]):
    # Get summary
    return summary(
        model=model,
        input_size=input_size,
        col_names=col_names,
        col_width=col_width,
        row_settings=row_settings
    )