from lib.convert import convert_to_coreml, inspect_model, make_updatable, inspect_model_instance
from lib.data import generate_random_2d_points
from lib.model import create_model


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    output_model = './exports/output.mlmodel'
    modifiable_output_model = './exports/output.mlmodel'

    data = generate_random_2d_points()
    model = create_model(data, data)

    mlmodel = convert_to_coreml(model)
    mlmodel.save(output_model)

    builder = inspect_model_instance(mlmodel)
    updatable = make_updatable(builder, output_model, modifiable_output_model)

    inspect_model(modifiable_output_model)