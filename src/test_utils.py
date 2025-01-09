from termcolor import colored

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense

# Compare the two inputs
def comparator(learner, instructor):
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    result = []
    for layer in model.layers:
        if hasattr(layer, 'output'):
            output_shape = layer.output.shape
        else:
            output_shape = None
            
        # Ensure consistency with expected summary format
        descriptors = [layer.__class__.__name__, [output_shape] if layer.__class__.__name__ == 'InputLayer' else output_shape, layer.count_params()]
        
        if isinstance(layer, Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if isinstance(layer, MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if isinstance(layer, Dropout):
            descriptors.append(layer.rate)
        if isinstance(layer, ZeroPadding2D):
            descriptors.append(layer.padding)
        if isinstance(layer, Dense):
            descriptors.append(layer.activation.__name__)
        result.append(descriptors)
    return result
