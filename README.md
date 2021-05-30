This package will help you to train and predict using CNN model

To use this package you'll need to create an object of base class
for example:
    
    obj = TrainCNNWithResnet(trainDatasetPath = None, validationDatasetPath = None, datasetFormat = "folder", 
                                imageSize = [224,224], useDatagenerator = True, loadModel = False)
                                
    @params
    trainDatasetPath      : Define path where training dataset available
    validationDatasetPath : Define path where testing dataset available
    datasetFormat         : Define format in which dataset available (folder, csv, json etc)
    imageSize             : Shape of the image (rows, column)
    useDatagenerator      : True if you want to use ImageDataGenerator to increase size of data else False
    loadModel             : True if you want to load previously trained model but it should be available at {modelName}\final-model.hdf5

Train the model :
    
    obj.train(epochs = 50, filepath = "Default"  )

    @params
    epochs   : define number of epoch
    filepath : to set new path for saving model weight file

Predict from model
    
    obj.predict(features, normalize = True)

    @params
    features : Image 
    normalize: to apply normalization
