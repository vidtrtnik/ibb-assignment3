import torch
from PIL import Image
from torchvision import transforms
import os
import sys
import time
import psutil

model = 0
times = []
mems = []

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles 
    
def classify(paths):
	global times
	global mems
	for path in paths:
		if(path.endswith(".jpg") == False):
			continue;
		print("File: " + path)
		input_image = Image.open(path)
		preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		input_tensor = preprocess(input_image)
		input_batch = input_tensor.unsqueeze(0)
		
		t1 = time.time()
		with torch.no_grad():
			scores = model(input_batch)
		t2 = time.time()
		mem = psutil.virtual_memory().used
		mems.append(mem)
		
		td = t2 - t1
		times.append(td)
		
		#print(scores.shape)

		with open('imagenet_classes.txt') as f:
		  classes = [line.strip() for line in f.readlines()]
		  
		_, index = torch.max(scores, 1)
		percentage = torch.nn.functional.softmax(scores, dim=1)[0] * 100
		_, indices = torch.sort(scores, descending=True)
		[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

		#print(classes[index[0]], percentage[index[0]].item())
		#print(torch.nn.functional.softmax(output[0], dim=0))
		result = classes[index[0]]
		probability = percentage[index[0]].item()
		
		logResult(path, result, probability)
		

def logResult(path, result, probability):
	global processed
	global correct 
	global averageProbability

	pathsplit = path.split('/')
	dset = pathsplit[2]
	dclass = pathsplit[3]
	fname = pathsplit[4]
	
	processed = processed + 1
	print("Predited (" + str(int(probability))+ "): " + result + ", actual: " + dclass + "\n")
	out = "X"
	if(result == dclass):
		correct = correct + 1
		out = "OK"
	print("Processed: " + str(processed) + ", " + "correct: " + str(correct) + "\n")
	
	
	if os.path.exists(resultsFolder + 'results_' + 'global' + '.txt'):
		append_write = 'a'
	else:
		append_write = 'w'
	with open(resultsFolder + 'results_' + 'global' + '.txt', append_write) as f:
		f.write(path + "\t###\t" + dset + "\t###\t" + dclass + "\t###\t" + result + "\t###\t" + str(int(probability)) + "\t###\t" + out + "\n")
		
		
	if os.path.exists(resultsFolder + 'results_' + dclass + '.txt'):
		append_write = 'a'
	else:
		append_write = 'w'		
	with open(resultsFolder + 'results_' + dclass + '.txt', append_write) as f:
		f.write(path + "\t###\t" + dset + "\t###\t" + dclass + "\t###\t" + result + "\t###\t" + str(int(probability)) + "\t###\t" + out + "\n")
	
	averageProbability = averageProbability + probability
	

averageProbability = 0
processed = 0
correct = 0 
resultsFolder=""
def main():
	global processed
	global correct 
	global averageProbability
	global resultsFolder
	global model
	
	if(len(sys.argv) < 3):
		return 0
		
	modelname = sys.argv[1]
	datasetpath = "./" + sys.argv[2]
	resultsFolder = 'results_' + modelname + '/'
	try:
		os.mkdir(resultsFolder)
	except Exception as e:
		dummy=0
		
	print("Model name: " + modelname + ", Dataset path: " + datasetpath + "\n")
	
	model = torch.hub.load('pytorch/vision:v0.5.0', modelname, pretrained=True)
	model.eval()
	
	dataset_list = getListOfFiles(datasetpath)
	classify(dataset_list)
	
	averageProbability = averageProbability / processed
	
	avgTime = 0
	for i in range(0, len(times)):
		avgTime += times[i]
	fullTime = avgTime
	avgTime = avgTime/len(times)
	
	avgMem = 0
	for i in range(0, len(mems)):
		avgMem += mems[i]
	avgMem = avgMem / len(mems)
	
	with open(resultsFolder + 'results_' + 'global' + '.txt', "a+") as f:
		  f.write("Processed: " + str(processed) + " Correct: " + str(correct) + "\n")
		  f.write("Average probability: " + str(averageProbability) + "\n")
		  f.write("Average time: " + str(avgTime) + "\n")
		  f.write("Full time: " + str(fullTime) + "\n")
		  f.write("Average memory: " + str(avgMem) + "\n")
main()
