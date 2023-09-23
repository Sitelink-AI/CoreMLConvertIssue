import coremltools
import coremltools as ct
from coremltools.models import MLModel, datatypes
from coremltools.models.neural_network import AdamParams
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType


def convert_to_coreml(tf_model):
    model = ct.convert(tf_model, source='tensorflow')
    # mlmodel = keras_converter.convert(keras_model, input_names=['image'],
    #                             output_names=['digitProbabilities'],
    #                             class_labels=class_labels,
    #                             predicted_feature_name='digit')

    return model


def inspect_model(coreml_model_path):
    spec = coremltools.utils.load_spec(coreml_model_path)
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
    builder.inspect_layers()
    return builder


def inspect_model_instance(coreml_model_instance):
    # Get the specification from the model instance
    spec = coreml_model_instance.get_spec()
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=spec)
    builder.inspect_layers(last=3)
    return builder

def make_updatable(builder, mlmodel_url, mlmodel_updatable_path):
    """This method makes an existing non-updatable mlmodel updatable.
    mlmodel_url - the path the Core ML model is stored.
    mlmodel_updatable_path - the path the updatable Core ML model will be saved.
    """

    model_spec = builder.spec

    # make_updatable method is used to make a layer updatable. It requires a list of layer names.
    # dense_1 and dense_2 are two innerProduct layer in this example and we make them updatable.
    builder.make_updatable(['sequential/dense_1/BiasAdd'])

    # model_spec.description.input[0].name = "xy_in"
    model_spec.description.input[
        0].shortDescription = 'The XY coordinate indicating where the system thinks a point is located'
    model_spec.description.output[0].shortDescription = 'The XY coordinate of the corrected output'

    # Categorical Cross Entropy or Mean Squared Error can be chosen for the loss layer.
    # Categorical Cross Entropy is used on this example. CCE requires two inputs: 'name' and 'input'.
    # name must be a string and will be the name associated with the loss layer
    # input must be the output of a softmax layer in the case of CCE.
    # The loss's target will be provided automatically as a part of the model's training inputs.
    # builder.set_categorical_cross_entropy_loss(name='lossLayer', input='digitProbabilities')

    # in addition of the loss layer, an optimizer must also be defined. SGD and Adam optimizers are supported.
    # SGD has been used for this example. To use SGD, one must set lr(learningRate) and batch(miniBatchSize) (momentum is an optional parameter).
    builder.set_adam_optimizer(AdamParams(lr=0.01, batch=32))

    feature = ('Identity', datatypes.Array(1, 2))
    builder.set_mean_squared_error_loss(name='lossLayer', input_feature=feature)

    # builder.set_training_input([('xy_input', datatypes.Array(2)), ('true_output', datatypes.Array(2))])

    # Finally, the number of epochs must be set as follows.
    builder.set_epochs(500)

    # builder.set_training_input('dense_input')

    # save the updated spec
    mlmodel_updatable = MLModel(model_spec)
    mlmodel_updatable.save(mlmodel_updatable_path)
    return mlmodel_updatable
