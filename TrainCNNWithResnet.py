import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from CNNModels.TrainModel import TrainModel
import numpy as np
from keras.preprocessing import image


class TrainCNNWithResnet(TrainModel):


    def __init__(self, trainDatasetPath = None, validationDatasetPath = None, datasetFormat = "folder", 
                                    imageSize = [224,224], useDatagenerator = True, loadModel = False):
        self.useDatagenerator = useDatagenerator
        
        super().__init__(trainDatasetPath, validationDatasetPath=validationDatasetPath, \
                                datasetFormat=datasetFormat, imageSize=imageSize, \
                                useDatagenerator=useDatagenerator, modelName = "Resnet-model")
        if loadModel:
            TrainModel.model = self.loadModel()
        else:
            TrainModel.model = self.getModel()
        
    
    def getModel(self):
        resnet = ResNet50(input_shape=self.imageSize + [3], weights='imagenet', include_top=False)
        # don't train existing weights
        for layer in resnet.layers:
            layer.trainable = False
        flattenLayer = Flatten()(resnet.output)
        prediction = Dense(self.nclasses, activation='softmax')(flattenLayer)
        # create a model object
        model = Model(inputs=resnet.input, outputs=prediction)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        return model

    

if __name__ == "__main__":
    trainDatasetPath = "Dataset\\Train"  
    testDatasetPath = "Dataset\\Test"
    obj = TrainCNNWithResnet(trainDatasetPath, testDatasetPath)
    obj.train(epochs=1)
    # loss, acc = obj.evaluate()
    # print(acc)
    # obj = TrainCNNWithResnet(loadModel = True)# trainDatasetPath, testDatasetPath)
    path = "F:\Machine Learning\car-brand-classification\Dataset\Test\\audi\\23.jpg"
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)
    # img = img/255.0
    img = np.expand_dims(img, axis=0)
    
    print(obj.predict(img, normalize = True))