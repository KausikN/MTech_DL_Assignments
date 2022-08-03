# cs6910_assignment1
CS6910 Deep Learning Assignment 1 Code

By,

Karthikeyan S (CS21M028)

N Kausik (CS21M037)

# Training
Command to train/test the model:
```shell
python main.py 
    --mode {mode} 
    --model {model}
    --dataset {dataset}
    --epochs {epochs}
    --batch_size {batch_size}
    --hidden_layers {hidden_layers}
    --hidden_neurons {hidden_neurons}
    --weight_decay {weight_decay}
    --optimiser {optimiser}
    --learning_rate {learning_rate}
    --gamma {gamma}
    --eps {eps}
    --beta1 {beta1}
    --beta2 {beta2}
    --init_func {init_func}
    --act_func {act_func}
    --loss_func {loss_func}
    --wandb
    --verbose
```

Parameters are,

    - mode: "train" or "test"
    - model:
        - save path of model for training
        - load path of model for testing
    - dataset:
        - "fashion" for fashion mnist
        - "mnist" for mnist
    - epochs: number of epochs
    - batch_size: batch size
    - hidden_layers: number of hidden layers
    - hidden_neurons: number of hidden neurons
    - weight_decay: weight decay
    - optimiser:
        - "momentum" for momentum optimiser
        - "sgd" for sgd optimiser
        - "nesterov" for nesterov optimiser
        - "adagrad" for AdaGrad optimiser
        - "rmsprop" for RMSprop optimiser
        - "adam" for Adam optimiser
        - "nadam" for Nadam optimiser
    - learning_rate: learning rate
    - gamma: gamma for momentum, nesterov, nadam
    - eps: eps for adagrad, rmsprop, adam, nadam
    - beta1: beta1 for adam, nadam
    - beta2: beta2 for rmsprop, adam, nadam
    - init_func:
        - "random" for Random initialisation
        - "xavier" for Xavier initialisation
    - act_func:
        - "relu" for ReLU activation
        - "tanh" for Tanh activation
        - "sigmoid" for Sigmoid activation
    - loss_func:
        - "cross_entropy" for Cross Entropy loss
        - "mse" for MSE loss
    - wandb:
        - for using wandb logging: Enter wandb details in config.json
    - verbose