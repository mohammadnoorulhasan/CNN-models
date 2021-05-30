
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from glob import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import os
from os.path import normpath
import json

class TrainModel:

    def __init__(self, trainDatasetPath = None, validationDatasetPath = None, datasetFormat = "folder",
                        imageSize = (224,224), useDatagenerator = True, modelName = "model"):
        
        self.modelName = modelName
        self.imageSize = imageSize
        self.finalModelPath = normpath(f"{self.modelName}/final-model.hdf5")
        self.modelConfigFile = normpath(f"{self.modelName}/model-config.json")
        
        if trainDatasetPath is not None:
            print("="*100)
            print("Loading Dataset")
            if datasetFormat == "folder":
                # We'll rescale while using Image Data Generator
                self._trainDatasetPath = normpath(trainDatasetPath)
                classes = glob(self._trainDatasetPath+"\\*")
                self.classes =[]
                for path in classes:
                    label = path.split("\\")[-1]
                    self.classes.append(label)
                print("Classes are :",self.classes)
                
                self.nclasses = len(classes)
                print
                if useDatagenerator:
                    self._trainDatagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)


                else:
                    self._trainDatagen = ImageDataGenerator(rescale = 1./255)

                self._trainDataset = self.loadFolderDataset(self._trainDatasetPath,self._trainDatagen)
                
                if validationDatasetPath is not None:
                    self._validationDatasetPath = validationDatasetPath
                    self._validationDatagen = ImageDataGenerator(rescale= 1.0/255)
                    self._validationDataset = self.loadFolderDataset(self._validationDatasetPath,
                                                                self._validationDatagen)

                    
                else:
                    self._validationDatagen = None
                print("Dataset loaded successfully")
            elif datasetFormat == "csv":
                pass

            else:
                print("wrong dataset format passed\n Please select \"folder\" or \csv\"")
            
            print("="*100)


    def loadFolderDataset(self,datasetPath, datagen, batchSize = 32, classMode = "categorical"):
        print(f"Loading dataset from : {datasetPath}")
        dataset = datagen.flow_from_directory( datasetPath,
                                                 target_size = self.imageSize,
                                                 batch_size = int(batchSize),
                                                 class_mode = classMode)
        
        return dataset

    def load_csv(self, datasetPath):
        pass


    def train(self,  useDatagenerator = True, epochs = 50, 
                        filepath = "Default" ):
        if filepath == "Default":
            filepath = self.modelName+"/training/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                                    verbose=1, save_best_only=True, mode='max') 
        
        callbacks_list = [checkpoint]
        print("="*100)
        print(f"Training of {self.modelName} start")
        print("="*100)
        print(self._trainDataset)
        if self._validationDatagen is not None:
            self._result = self.model.fit(self._trainDataset,
                                    validation_data=self._validationDataset,
                                    epochs=epochs,
                                    steps_per_epoch=int(len(self._trainDataset)),
                                    validation_steps=len(self._validationDataset),
                                    callbacks = callbacks_list)
            
        else:
            self.__result = self.model.fit(self._trainDataset,
                                    epochs=epochs,
                                    steps_per_epoch=len(self._trainDataset))


        epoch = 1+ np.argmax(self._result.history["val_accuracy"])
        accuracy = self._result.history["accuracy"][epoch-1]
        loss = self._result.history["loss"][epoch-1]
        val_accuracy = max(self._result.history["val_accuracy"])
        val_loss = self._result.history["val_loss"][epoch-1]
        
        self.trainModelPath = normpath(f"{self.modelName}/training/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
        print("="*100)
        print("Model Training Completed, below are the model accuracy measurements")
        print("+"*100)
        print("Training Accuracy :", accuracy)
        print("Training loss :", loss)
        print("Validation Accuracy :", val_accuracy)
        print("Validation loss :", val_loss)
        print("="*100)
        self.model = load_model(self.trainModelPath)
        self.model.save(self.finalModelPath)

        self.modelConfiguration = { "training-accuracy":accuracy,
                                "trainig-loss" : loss,
                                "val-accuracy" : val_accuracy,
                                "val-loss" : val_loss,
                                "n-classes" : self.nclasses,
                                "classes" : self.classes,
                                "image-size" : self.imageSize }
        with open(self.modelConfigFile , "w") as file:
            json.dump(self.modelConfiguration, file)
        self.classes = np.array(self.classes)


    def loadModel(self, modelPath = None):
        with open(self.modelConfigFile, "r") as file:
            self.modelConfiguration = json.load(file)
        
        self.nclasses = self.modelConfiguration["n-classes"]
        self.classes = self.modelConfiguration["classes"]
        self.classes = np.array(self.classes)
        self.imageSize = self.modelConfiguration["image-size"]
        
        if modelPath is None:
            self.model = load_model(self.finalModelPath)  
            print("Model Loaded")  
        else:
            self.model = load_model(modelPath)

    def evaluate(self):
        return self.model.evaluate(self._validationDataset, verbose = 0)

    def predictClass(self, prediction):
        pass
    
    def predict(self, features, normalize = True):
        if normalize:
            features = features/255

        predictions = self.model.predict(features)
        predictions = np.argmax(predictions, axis = 1)
        func = lambda predict : self.classes[predict]
        predictions = func(predictions)
        return predictions


    def isPathExist(self, path):
        if not os.path.exists(path):
            return False
        else: 
            return True
            