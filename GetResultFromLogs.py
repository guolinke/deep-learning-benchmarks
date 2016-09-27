FCN_batch_size = 64
CNN_batch_size = 16
RNN_batch_size = 128

def GetTimeStamp(line):
	full_time = line.split(' ')[1]
	mil =  float('0.'+full_time.split('.')[1])
	tokens = full_time.split('.')[0].split(':')
	hour = int(tokens[0])
	minute = int(tokens[1])
	second = int(tokens[2])
	return mil + second + minute * 60 + hour * 60 * 60

def GetTimeFromCaffeLog(filename):
	file_in = open(filename,"r")
	batch_size = -1
	if 'fcn' in filename:
		batch_size = FCN_batch_size
	elif 'lstm' in filename:
		batch_size = RNN_batch_size
	else:
		batch_size = CNN_batch_size

	start_time = 0
	start_iter = -1
	end_time = 0
	end_iter = 0

	max_iter = -1
	for line in file_in.readlines():
		if 'max_iter' in line and max_iter == -1:
			max_iter = int( line.split(':')[1] )
		if 'sgd_solver.cpp' in line and 'Iteration' in line:
			cur_iter = int(line.split('Iteration')[1].strip().split(',')[0])
			if start_iter == -1 and cur_iter > 0:
				start_iter = cur_iter
				start_time = GetTimeStamp(line)
			elif cur_iter > 0 and cur_iter < max_iter:
				end_iter = cur_iter
				end_time = GetTimeStamp(line)
	delta_time = end_time - start_time
	if delta_time < 0.0:
		delta_time += 24 * 60 * 60.0
	time = (end_time - start_time) / (end_iter - start_iter)
	sps = 1.0 / time * batch_size
	return [time, sps]
def GetCaffeResult():
	return [GetTimeFromCaffeLog("caffe/output_fcn5.log"),
	GetTimeFromCaffeLog("caffe/output_fcn8.log"),
	GetTimeFromCaffeLog("caffe/output_alexnet.log"),
	GetTimeFromCaffeLog("caffe/output_resnet.log"),
	[0,0],[0,0]]

def GetTimeFromCNTKLog(filename):
	file_in = open(filename,"r")
	epoch_size = -1
	mini_batch_size = -1
	for line in file_in.readlines():
		# get num_batches
		if 'epochSize' in line and len(line.split('=')) == 2:
			epoch_size = int(line.split('=')[1])
		if 'minibatchSize' in line and len(line.split('=')) == 2:
			mini_batch_size = int(line.split('=')[1])
		if 'Finished Epoch' in line:
			epo_info = line.split(":")[0].split('[')[1].split(']')[0]
			cur_epo = int(epo_info.split('of')[0])
			max_epo = int(epo_info.split('of')[1])
			if cur_epo == max_epo:
				epo_time = float( line.split(';')[-1].split('=')[1].split('s')[0] )
				num_batches =  epoch_size // mini_batch_size
				return[epo_time / num_batches, epoch_size / epo_time]
def GetCNTKResult():
	return [GetTimeFromCNTKLog("cntk/output_fcn5_Train.log"),
	GetTimeFromCNTKLog("cntk/output_fcn8_Train.log"),
	GetTimeFromCNTKLog("cntk/output_alexnet_Train.log"),
	GetTimeFromCNTKLog("cntk/output_resnet_Train.log"),
	GetTimeFromCNTKLog("cntk/output_lstm32_Train.log"),
	GetTimeFromCNTKLog("cntk/output_lstm64_Train.log")]

def GetTimeFromTensorflowLog(filename):
	file_in = open(filename,"r")
	if 'fcn' in filename:
		sps = -1
		time = -1
		for line in file_in.readlines():
			if 'Training speed (samples/sec)' in line:
				sps = int(line.split(':')[1].split(',')[0].split('=')[1])
			if 'fc across' in line:
				time = float( line.split(',')[-1].split('sec')[0])
		return [time, sps]
	elif 'lstm' in filename:
		batch_size = -1
		for line in file_in.readlines():
			if 'batch_size:' in line:
				batch_size = int(line.split(':')[-1])
			if 'for one mini batch' in line:
				time = float( line.split('seconds.')[-1].split('seconds ')[0])
		if '32' in filename:
			return [time, 32.0 / time * batch_size]
		else:
			return [time, 64.0 / time * batch_size]
	elif 'resnet' in filename:
		total_samples = 0
		total_time = 0
		num_batches = 0
		for line in file_in.readlines():
			if 'step' in line and not 'step 0' in line:
				tokens = line.split('(')[-1].split(')')[0]
				time_str = tokens.split(';')[1].strip()
				sps_str = tokens.split(';')[0].strip()
				num_batches += 1
				time = float(time_str.split(' ')[0])
				total_time += time
				cur_samples = float(sps_str.split(' ')[0]) * time
				total_samples += cur_samples
		return [total_time / num_batches, total_samples / total_time]
	else:
		for line in file_in.readlines():
			if 'Forward-backward across' in line:
				time = float(line.split(',')[-1].split('+/-')[0])
				return [time, CNN_batch_size * 1.0 / time]	
def GetTersonflowResult():
	return[GetTimeFromTensorflowLog('tensorflow/output_fcn5.log'),
	GetTimeFromTensorflowLog('tensorflow/output_fcn8.log'),
	GetTimeFromTensorflowLog('tensorflow/output_alexnet.log'),
	GetTimeFromTensorflowLog('tensorflow/output_resnet.log'),
	GetTimeFromTensorflowLog('tensorflow/output_lstm32.log'),
	GetTimeFromTensorflowLog('tensorflow/output_lstm64.log')]


def GetTimeFromTheanoLog(filename):
	file_in = open(filename,"r")
	batch_size = -1
	if 'alexnet' in filename or 'resnet' in filename:
		batch_size = CNN_batch_size
	elif 'lstm' in filename:
		batch_size = RNN_batch_size
	else:
		batch_size = FCN_batch_size
	for line in file_in.readlines():
		if 'Forward-Backward across' in line:
			time = float(line.split(',')[-1].split('+/-')[0])
	if 'lstm32' in filename:
		return [time, batch_size * 32.0 / time]
	elif 'lstm64' in filename:
		return [time, batch_size * 64.0 / time]
	else:
		return [time, batch_size / time]
def GetTheanoResult():
	return [GetTimeFromTheanoLog('theano/log/fcn5.log'),
	GetTimeFromTheanoLog('theano/log/fcn8.log'),
	GetTimeFromTheanoLog('theano/log/alexnet.log'),
	GetTimeFromTheanoLog('theano/log/resnet.log'),
	GetTimeFromTheanoLog('theano/log/lstm32.log'),
	GetTimeFromTheanoLog('theano/log/lstm64.log')]

def GetTimeFromTorchLog(filename):
	file_in = open(filename,"r")
	batch_size = -1
	if 'alexnet' in filename or 'resnet' in filename:
		batch_size = CNN_batch_size
	elif 'lstm' in filename:
		batch_size = RNN_batch_size
	else:
		batch_size = FCN_batch_size
	if 'lstm' in filename:
		for line in file_in.readlines():
			if 'Time elapsed for' in line:
				it = int(line.split('iters:')[0].strip().split(' ')[-1])
				time = float(line.split('iters:')[1].strip().split(' ')[0])
				time = time / it
		if 'lstm32' in filename:
			return [time, batch_size * 32.0 / time]
		else:
			return [time, batch_size * 64.0 / time]
	else:	
		for line in file_in.readlines():
			if '(sec/mini-batch)' in line:
				time = float(line.split(':')[-1])
				return [time, batch_size * 1.0 / time]

def GetTorchResult():
	return[GetTimeFromTorchLog('torch7/output_fcn5.log'),
	GetTimeFromTorchLog('torch7/output_fcn8.log'),
	GetTimeFromTorchLog('torch7/output_alexnet.log'),
	GetTimeFromTorchLog('torch7/output_resnet.log'),
	GetTimeFromTorchLog('torch7/output_lstm32.log'),
	GetTimeFromTorchLog('torch7/output_lstm64.log')]

def GetTimeFromMxnetLog(filename):
	file_in = open(filename,"r")
	batch_size = -1
	if 'alexnet' in filename or 'resnet' in filename:
		batch_size = CNN_batch_size
	elif 'lstm' in filename:
		batch_size = RNN_batch_size
	else:
		batch_size = FCN_batch_size
	if 'lstm' in filename:
		for line in file_in.readlines():
			if 'Batch [50]' in line:
				sps = float(line.split('Speed:')[1].split('samples/sec')[0])
		if 'lstm32' in filename:
			return [batch_size * 32.0 / sps, sps]
		else:
			return [batch_size * 64.0 / sps, sps]
	else:	
		for line in file_in.readlines():
			if '(sec/mini-batch)' in line:
				time = float(line.split(':')[-1])
				return [time, batch_size * 1.0 / time]
def GetMxnetResult():
	return[GetTimeFromMxnetLog('mxnet/output_fcn5.log'),
	GetTimeFromMxnetLog('mxnet/output_fcn8.log'),
	GetTimeFromMxnetLog('mxnet/output_alexnet.log'),
	GetTimeFromMxnetLog('mxnet/output_resnet.log'),
	GetTimeFromMxnetLog('mxnet/output_lstm32.log'),
	GetTimeFromMxnetLog('mxnet/output_lstm64.log')]

caffe_result = GetCaffeResult()
cntk_result = GetCNTKResult()
mxnet_resutl = GetMxnetResult()
tf_result = GetTersonflowResult()
th_result = GetTheanoResult()
to_result = GetTorchResult()
names = ['Caffe','CNTK', 'MXNet' ,'TensorFlow','Theano','Torch']
result = [caffe_result, cntk_result, mxnet_resutl, tf_result, th_result, to_result] 
file_out = open('result.md','w')
file_out.write('seconds/num_batches:\n\n')
file_out.write('| Tool | FCN-5 | FCN-8 | AlexNet | ResNet | LSTM-32 | LSTM-64 |\n')
file_out.write('|------|-------|-------|---------|--------|---------|---------|\n')
for i in xrange(len(names)):
	file_out.write('|' + names[i])
	for x in result[i]:
		file_out.write('| %.3f ' %(x[0]) )
	file_out.write('|\n')

file_out.write('\n\nsamples/second:\n\n')
file_out.write('| Tool | FCN-5 | FCN-8 | AlexNet | ResNet | LSTM-32 | LSTM-64 |\n')
file_out.write('|------|-------|-------|---------|--------|---------|---------|\n')
for i in xrange(len(names)):
	file_out.write('|' + names[i])
	for x in result[i]:
		file_out.write('| %d ' %(x[1]) )
	file_out.write('|\n')
file_out.close()
