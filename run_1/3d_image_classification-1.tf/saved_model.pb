ѓЙ
юЌ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
÷
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

$
DisableCopyOnRead
resourceИ
ы
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
ј
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
П
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ПО
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/v/dense_1/kernel
А
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	А*
dtype0
З
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/m/dense_1/kernel
А
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	А*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:А*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:А*
dtype0
Д
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
АА*
dtype0
Д
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
АА*
dtype0
Ы
!Adam/v/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/v/batch_normalization_3/beta
Ф
5Adam/v/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Ы
!Adam/m/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/m/batch_normalization_3/beta
Ф
5Adam/m/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Э
"Adam/v/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/batch_normalization_3/gamma
Ц
6Adam/v/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Э
"Adam/m/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/batch_normalization_3/gamma
Ц
6Adam/m/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Б
Adam/v/conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv3d_3/bias
z
(Adam/v/conv3d_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_3/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv3d_3/bias
z
(Adam/m/conv3d_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_3/bias*
_output_shapes	
:А*
dtype0
Ц
Adam/v/conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА*'
shared_nameAdam/v/conv3d_3/kernel
П
*Adam/v/conv3d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_3/kernel*,
_output_shapes
:АА*
dtype0
Ц
Adam/m/conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА*'
shared_nameAdam/m/conv3d_3/kernel
П
*Adam/m/conv3d_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_3/kernel*,
_output_shapes
:АА*
dtype0
Ы
!Adam/v/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/v/batch_normalization_2/beta
Ф
5Adam/v/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_2/beta*
_output_shapes	
:А*
dtype0
Ы
!Adam/m/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!Adam/m/batch_normalization_2/beta
Ф
5Adam/m/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_2/beta*
_output_shapes	
:А*
dtype0
Э
"Adam/v/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/v/batch_normalization_2/gamma
Ц
6Adam/v/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
Э
"Adam/m/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/m/batch_normalization_2/gamma
Ц
6Adam/m/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
Б
Adam/v/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/v/conv3d_2/bias
z
(Adam/v/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/bias*
_output_shapes	
:А*
dtype0
Б
Adam/m/conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/m/conv3d_2/bias
z
(Adam/m/conv3d_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/bias*
_output_shapes	
:А*
dtype0
Х
Adam/v/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*'
shared_nameAdam/v/conv3d_2/kernel
О
*Adam/v/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_2/kernel*+
_output_shapes
:@А*
dtype0
Х
Adam/m/conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А*'
shared_nameAdam/m/conv3d_2/kernel
О
*Adam/m/conv3d_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_2/kernel*+
_output_shapes
:@А*
dtype0
Ъ
!Adam/v/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/v/batch_normalization_1/beta
У
5Adam/v/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ъ
!Adam/m/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/m/batch_normalization_1/beta
У
5Adam/m/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
Ь
"Adam/v/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/v/batch_normalization_1/gamma
Х
6Adam/v/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
Ь
"Adam/m/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/m/batch_normalization_1/gamma
Х
6Adam/m/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
А
Adam/v/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv3d_1/bias
y
(Adam/v/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/bias*
_output_shapes
:@*
dtype0
А
Adam/m/conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv3d_1/bias
y
(Adam/m/conv3d_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/bias*
_output_shapes
:@*
dtype0
Ф
Adam/v/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/v/conv3d_1/kernel
Н
*Adam/v/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d_1/kernel**
_output_shapes
:@@*
dtype0
Ф
Adam/m/conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/m/conv3d_1/kernel
Н
*Adam/m/conv3d_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d_1/kernel**
_output_shapes
:@@*
dtype0
Ц
Adam/v/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/v/batch_normalization/beta
П
3Adam/v/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/v/batch_normalization/beta*
_output_shapes
:@*
dtype0
Ц
Adam/m/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!Adam/m/batch_normalization/beta
П
3Adam/m/batch_normalization/beta/Read/ReadVariableOpReadVariableOpAdam/m/batch_normalization/beta*
_output_shapes
:@*
dtype0
Ш
 Adam/v/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/v/batch_normalization/gamma
С
4Adam/v/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/v/batch_normalization/gamma*
_output_shapes
:@*
dtype0
Ш
 Adam/m/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/m/batch_normalization/gamma
С
4Adam/m/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp Adam/m/batch_normalization/gamma*
_output_shapes
:@*
dtype0
|
Adam/v/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/v/conv3d/bias
u
&Adam/v/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/bias*
_output_shapes
:@*
dtype0
|
Adam/m/conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/m/conv3d/bias
u
&Adam/m/conv3d/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/bias*
_output_shapes
:@*
dtype0
Р
Adam/v/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/conv3d/kernel
Й
(Adam/v/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv3d/kernel**
_output_shapes
:@*
dtype0
Р
Adam/m/conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/conv3d/kernel
Й
(Adam/m/conv3d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv3d/kernel**
_output_shapes
:@*
dtype0
~
current_learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namecurrent_learning_rate
w
)current_learning_rate/Read/ReadVariableOpReadVariableOpcurrent_learning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АА*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
s
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv3d_3/bias
l
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes	
:А*
dtype0
И
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:АА* 
shared_nameconv3d_3/kernel
Б
#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel*,
_output_shapes
:АА*
dtype0
£
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_2/moving_variance
Ь
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_2/moving_mean
Ф
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes	
:А*
dtype0
Н
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_2/beta
Ж
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes	
:А*
dtype0
П
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_2/gamma
И
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes	
:А*
dtype0
s
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv3d_2/bias
l
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes	
:А*
dtype0
З
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@А* 
shared_nameconv3d_2/kernel
А
#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel*+
_output_shapes
:@А*
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:@*
dtype0
Ж
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:@@*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:@*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:@*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:@*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:@*
dtype0
В
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:@*
dtype0
Ц
serving_default_input_1Placeholder*5
_output_shapes#
!:€€€€€€€€€АА@*
dtype0**
shape!:€€€€€€€€€АА@
•
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv3d_3/kernelconv3d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_278440

NoOpNoOp
”Ю
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*НЮ
valueВЮBюЭ BцЭ
і
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op*
О
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
’
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis
	1gamma
2beta
3moving_mean
4moving_variance*
»
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op*
О
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
’
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*
»
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*
О
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
’
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance*
»
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
 q_jit_compiled_convolution_op*
О
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
Ў
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance*
Ф
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses* 
Ѓ
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
Пkernel
	Рbias*
ђ
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator* 
Ѓ
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias*
б
!0
"1
12
23
34
45
;6
<7
K8
L9
M10
N11
U12
V13
e14
f15
g16
h17
o18
p19
20
А21
Б22
В23
П24
Р25
Ю26
Я27*
Я
!0
"1
12
23
;4
<5
K6
L7
U8
V9
e10
f11
o12
p13
14
А15
П16
Р17
Ю18
Я19*
* 
µ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

•trace_0
¶trace_1* 

Іtrace_0
®trace_1* 
* 
Р
©
_variables
™_iterations
Ђ_current_learning_rate
ђ_index_dict
≠
_momentums
Ѓ_velocities
ѓ_update_step_xla*

∞serving_default* 

!0
"1*

!0
"1*
* 
Ш
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

ґtrace_0* 

Јtrace_0* 
]W
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

љtrace_0* 

Њtrace_0* 
 
10
21
32
43*

10
21*
* 
Ш
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

ƒtrace_0
≈trace_1* 

∆trace_0
«trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
Ш
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Ќtrace_0* 

ќtrace_0* 
_Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

‘trace_0* 

’trace_0* 
 
K0
L1
M2
N3*

K0
L1*
* 
Ш
÷non_trainable_variables
„layers
Ўmetrics
 ўlayer_regularization_losses
Џlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

џtrace_0
№trace_1* 

Ёtrace_0
ёtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
* 
Ш
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

дtrace_0* 

еtrace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

лtrace_0* 

мtrace_0* 
 
e0
f1
g2
h3*

e0
f1*
* 
Ш
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

тtrace_0
уtrace_1* 

фtrace_0
хtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

o0
p1*
* 
Ш
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

ыtrace_0* 

ьtrace_0* 
_Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

Вtrace_0* 

Гtrace_0* 
#
0
А1
Б2
В3*

0
А1*
* 
Ш
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

Йtrace_0
Кtrace_1* 

Лtrace_0
Мtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses* 

Тtrace_0* 

Уtrace_0* 

П0
Р1*

П0
Р1*
* 
Ю
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses*

Щtrace_0* 

Ъtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

†trace_0
°trace_1* 

Ґtrace_0
£trace_1* 
* 

Ю0
Я1*

Ю0
Я1*
* 
Ю
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses*

©trace_0* 

™trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
>
30
41
M2
N3
g4
h5
Б6
В7*
В
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

Ђ0
ђ1*
* 
* 
* 
* 
* 
* 
л
™0
≠1
Ѓ2
ѓ3
∞4
±5
≤6
≥7
і8
µ9
ґ10
Ј11
Є12
є13
Ї14
ї15
Љ16
љ17
Њ18
њ19
ј20
Ѕ21
¬22
√23
ƒ24
≈25
∆26
«27
»28
…29
 30
Ћ31
ћ32
Ќ33
ќ34
ѕ35
–36
—37
“38
”39
‘40*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ѓ
≠0
ѓ1
±2
≥3
µ4
Ј5
є6
ї7
љ8
њ9
Ѕ10
√11
≈12
«13
…14
Ћ15
Ќ16
ѕ17
—18
”19*
Ѓ
Ѓ0
∞1
≤2
і3
ґ4
Є5
Ї6
Љ7
Њ8
ј9
¬10
ƒ11
∆12
»13
 14
ћ15
ќ16
–17
“18
‘19*
§
’trace_0
÷trace_1
„trace_2
Ўtrace_3
ўtrace_4
Џtrace_5
џtrace_6
№trace_7
Ёtrace_8
ёtrace_9
яtrace_10
аtrace_11
бtrace_12
вtrace_13
гtrace_14
дtrace_15
еtrace_16
жtrace_17
зtrace_18
иtrace_19* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

30
41*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

g0
h1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Б0
В1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
й	variables
к	keras_api

лtotal

мcount*
M
н	variables
о	keras_api

пtotal

рcount
с
_fn_kwargs*
_Y
VARIABLE_VALUEAdam/m/conv3d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv3d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/conv3d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/conv3d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv3d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv3d_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv3d_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv3d_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv3d_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_3/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_3/gamma2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_3/beta2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_3/beta2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

л0
м1*

й	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

п0
р1*

н	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ґ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv3d_3/kernelconv3d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationcurrent_learning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv3d_3/kernelAdam/v/conv3d_3/kernelAdam/m/conv3d_3/biasAdam/v/conv3d_3/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcountConst*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_279352
Э
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv3d_1/kernelconv3d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv3d_2/kernelconv3d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv3d_3/kernelconv3d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationcurrent_learning_rateAdam/m/conv3d/kernelAdam/v/conv3d/kernelAdam/m/conv3d/biasAdam/v/conv3d/bias Adam/m/batch_normalization/gamma Adam/v/batch_normalization/gammaAdam/m/batch_normalization/betaAdam/v/batch_normalization/betaAdam/m/conv3d_1/kernelAdam/v/conv3d_1/kernelAdam/m/conv3d_1/biasAdam/v/conv3d_1/bias"Adam/m/batch_normalization_1/gamma"Adam/v/batch_normalization_1/gamma!Adam/m/batch_normalization_1/beta!Adam/v/batch_normalization_1/betaAdam/m/conv3d_2/kernelAdam/v/conv3d_2/kernelAdam/m/conv3d_2/biasAdam/v/conv3d_2/bias"Adam/m/batch_normalization_2/gamma"Adam/v/batch_normalization_2/gamma!Adam/m/batch_normalization_2/beta!Adam/v/batch_normalization_2/betaAdam/m/conv3d_3/kernelAdam/v/conv3d_3/kernelAdam/m/conv3d_3/biasAdam/v/conv3d_3/bias"Adam/m/batch_normalization_3/gamma"Adam/v/batch_normalization_3/gamma!Adam/m/batch_normalization_3/beta!Adam/v/batch_normalization_3/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcount*V
TinO
M2K*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_279583ц§
ў
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_278746

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≠
Ґ
)__inference_conv3d_1_layer_call_fn_278541

inputs%
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€==@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_277996{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€==@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€??@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278537:&"
 
_user_specified_name278535:[ W
3
_output_shapes!
:€€€€€€€€€??@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29265
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
„
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_277663

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
’
Њ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277686

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
б
\
"__inference__update_step_xla_29315
gradient(
variable:АА*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*-
_input_shapes
:АА: *
	_noinline(:($
"
_user_specified_name
variable:V R
,
_output_shapes
:АА
"
_user_specified_name
gradient
€
В
$__inference_signature_wrapper_278440
input_1%
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@'
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@)

unknown_11:@А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А*

unknown_17:АА

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_277658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278436:&"
 
_user_specified_name278434:&"
 
_user_specified_name278432:&"
 
_user_specified_name278430:&"
 
_user_specified_name278428:&"
 
_user_specified_name278426:&"
 
_user_specified_name278424:&"
 
_user_specified_name278422:&"
 
_user_specified_name278420:&"
 
_user_specified_name278418:&"
 
_user_specified_name278416:&"
 
_user_specified_name278414:&"
 
_user_specified_name278412:&"
 
_user_specified_name278410:&"
 
_user_specified_name278408:&"
 
_user_specified_name278406:&"
 
_user_specified_name278404:&"
 
_user_specified_name278402:&
"
 
_user_specified_name278400:&	"
 
_user_specified_name278398:&"
 
_user_specified_name278396:&"
 
_user_specified_name278394:&"
 
_user_specified_name278392:&"
 
_user_specified_name278390:&"
 
_user_specified_name278388:&"
 
_user_specified_name278386:&"
 
_user_specified_name278384:&"
 
_user_specified_name278382:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
з
ƒ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278698

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
”
p
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_277952

inputs
identityk
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≠
†
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277848

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29290
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
НY
Ж
A__inference_3dcnn_layer_call_and_return_conditional_losses_278111
input_1+
conv3d_277971:@
conv3d_277973:@(
batch_normalization_277977:@(
batch_normalization_277979:@(
batch_normalization_277981:@(
batch_normalization_277983:@-
conv3d_1_277997:@@
conv3d_1_277999:@*
batch_normalization_1_278003:@*
batch_normalization_1_278005:@*
batch_normalization_1_278007:@*
batch_normalization_1_278009:@.
conv3d_2_278023:@А
conv3d_2_278025:	А+
batch_normalization_2_278029:	А+
batch_normalization_2_278031:	А+
batch_normalization_2_278033:	А+
batch_normalization_2_278035:	А/
conv3d_3_278049:АА
conv3d_3_278051:	А+
batch_normalization_3_278055:	А+
batch_normalization_3_278057:	А+
batch_normalization_3_278059:	А+
batch_normalization_3_278061:	А 
dense_278076:
АА
dense_278078:	А!
dense_1_278105:	А
dense_1_278107:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdropout/StatefulPartitionedCallш
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_277971conv3d_277973*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€~~>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_277970т
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€??@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_277663Е
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0batch_normalization_277977batch_normalization_277979batch_normalization_277981batch_normalization_277983*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€??@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277686≠
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_277997conv3d_1_277999*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€==@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_277996ш
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_277735У
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0batch_normalization_1_278003batch_normalization_1_278005batch_normalization_1_278007batch_normalization_1_278009*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277758∞
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_278023conv3d_2_278025*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278022щ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_277807Ф
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0batch_normalization_2_278029batch_normalization_2_278031batch_normalization_2_278033batch_normalization_2_278035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277830∞
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_278049conv3d_3_278051*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278048щ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_277879Ф
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0batch_normalization_3_278055batch_normalization_3_278057batch_normalization_3_278059batch_normalization_3_278061*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277902М
(global_average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_277952У
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_278076dense_278078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_278075к
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_278092С
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_278105dense_1_278107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_278104w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ќ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:&"
 
_user_specified_name278107:&"
 
_user_specified_name278105:&"
 
_user_specified_name278078:&"
 
_user_specified_name278076:&"
 
_user_specified_name278061:&"
 
_user_specified_name278059:&"
 
_user_specified_name278057:&"
 
_user_specified_name278055:&"
 
_user_specified_name278051:&"
 
_user_specified_name278049:&"
 
_user_specified_name278035:&"
 
_user_specified_name278033:&"
 
_user_specified_name278031:&"
 
_user_specified_name278029:&"
 
_user_specified_name278025:&"
 
_user_specified_name278023:&"
 
_user_specified_name278009:&"
 
_user_specified_name278007:&
"
 
_user_specified_name278005:&	"
 
_user_specified_name278003:&"
 
_user_specified_name277999:&"
 
_user_specified_name277997:&"
 
_user_specified_name277983:&"
 
_user_specified_name277981:&"
 
_user_specified_name277979:&"
 
_user_specified_name277977:&"
 
_user_specified_name277973:&"
 
_user_specified_name277971:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
Ю

b
C__inference_dropout_layer_call_and_return_conditional_losses_278861

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
џ
Z
"__inference__update_step_xla_29255
gradient&
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:@: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:@
"
_user_specified_name
gradient
Ы
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277704

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29260
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
ћ

’
6__inference_batch_normalization_3_layer_call_fn_278772

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277920Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278768:&"
 
_user_specified_name278766:&"
 
_user_specified_name278764:&"
 
_user_specified_name278762:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Џ
a
C__inference_dropout_layer_call_and_return_conditional_losses_278184

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
 

’
6__inference_batch_normalization_3_layer_call_fn_278759

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277902Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278755:&"
 
_user_specified_name278753:&"
 
_user_specified_name278751:&"
 
_user_specified_name278749:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ѓ
K
"__inference__update_step_xla_29300
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
Ѓ
K
"__inference__update_step_xla_29320
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
‘

х
A__inference_dense_layer_call_and_return_conditional_losses_278839

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Э
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278624

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29280
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
ў
€
B__inference_conv3d_layer_call_and_return_conditional_losses_278460

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€~~>@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:€€€€€€€€€АА@
 
_user_specified_nameinputs
≠
†
'__inference_conv3d_layer_call_fn_278449

inputs%
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€~~>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_277970{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€~~>@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278445:&"
 
_user_specified_name278443:] Y
5
_output_shapes#
!:€€€€€€€€€АА@
 
_user_specified_nameinputs
„
ј
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277758

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
п
L
0__inference_max_pooling3d_3_layer_call_fn_278741

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_277879Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
йЏ
Ж1
"__inference__traced_restore_279583
file_prefix<
assignvariableop_conv3d_kernel:@,
assignvariableop_1_conv3d_bias:@:
,assignvariableop_2_batch_normalization_gamma:@9
+assignvariableop_3_batch_normalization_beta:@@
2assignvariableop_4_batch_normalization_moving_mean:@D
6assignvariableop_5_batch_normalization_moving_variance:@@
"assignvariableop_6_conv3d_1_kernel:@@.
 assignvariableop_7_conv3d_1_bias:@<
.assignvariableop_8_batch_normalization_1_gamma:@;
-assignvariableop_9_batch_normalization_1_beta:@C
5assignvariableop_10_batch_normalization_1_moving_mean:@G
9assignvariableop_11_batch_normalization_1_moving_variance:@B
#assignvariableop_12_conv3d_2_kernel:@А0
!assignvariableop_13_conv3d_2_bias:	А>
/assignvariableop_14_batch_normalization_2_gamma:	А=
.assignvariableop_15_batch_normalization_2_beta:	АD
5assignvariableop_16_batch_normalization_2_moving_mean:	АH
9assignvariableop_17_batch_normalization_2_moving_variance:	АC
#assignvariableop_18_conv3d_3_kernel:АА0
!assignvariableop_19_conv3d_3_bias:	А>
/assignvariableop_20_batch_normalization_3_gamma:	А=
.assignvariableop_21_batch_normalization_3_beta:	АD
5assignvariableop_22_batch_normalization_3_moving_mean:	АH
9assignvariableop_23_batch_normalization_3_moving_variance:	А4
 assignvariableop_24_dense_kernel:
АА-
assignvariableop_25_dense_bias:	А5
"assignvariableop_26_dense_1_kernel:	А.
 assignvariableop_27_dense_1_bias:'
assignvariableop_28_iteration:	 3
)assignvariableop_29_current_learning_rate: F
(assignvariableop_30_adam_m_conv3d_kernel:@F
(assignvariableop_31_adam_v_conv3d_kernel:@4
&assignvariableop_32_adam_m_conv3d_bias:@4
&assignvariableop_33_adam_v_conv3d_bias:@B
4assignvariableop_34_adam_m_batch_normalization_gamma:@B
4assignvariableop_35_adam_v_batch_normalization_gamma:@A
3assignvariableop_36_adam_m_batch_normalization_beta:@A
3assignvariableop_37_adam_v_batch_normalization_beta:@H
*assignvariableop_38_adam_m_conv3d_1_kernel:@@H
*assignvariableop_39_adam_v_conv3d_1_kernel:@@6
(assignvariableop_40_adam_m_conv3d_1_bias:@6
(assignvariableop_41_adam_v_conv3d_1_bias:@D
6assignvariableop_42_adam_m_batch_normalization_1_gamma:@D
6assignvariableop_43_adam_v_batch_normalization_1_gamma:@C
5assignvariableop_44_adam_m_batch_normalization_1_beta:@C
5assignvariableop_45_adam_v_batch_normalization_1_beta:@I
*assignvariableop_46_adam_m_conv3d_2_kernel:@АI
*assignvariableop_47_adam_v_conv3d_2_kernel:@А7
(assignvariableop_48_adam_m_conv3d_2_bias:	А7
(assignvariableop_49_adam_v_conv3d_2_bias:	АE
6assignvariableop_50_adam_m_batch_normalization_2_gamma:	АE
6assignvariableop_51_adam_v_batch_normalization_2_gamma:	АD
5assignvariableop_52_adam_m_batch_normalization_2_beta:	АD
5assignvariableop_53_adam_v_batch_normalization_2_beta:	АJ
*assignvariableop_54_adam_m_conv3d_3_kernel:ААJ
*assignvariableop_55_adam_v_conv3d_3_kernel:АА7
(assignvariableop_56_adam_m_conv3d_3_bias:	А7
(assignvariableop_57_adam_v_conv3d_3_bias:	АE
6assignvariableop_58_adam_m_batch_normalization_3_gamma:	АE
6assignvariableop_59_adam_v_batch_normalization_3_gamma:	АD
5assignvariableop_60_adam_m_batch_normalization_3_beta:	АD
5assignvariableop_61_adam_v_batch_normalization_3_beta:	А;
'assignvariableop_62_adam_m_dense_kernel:
АА;
'assignvariableop_63_adam_v_dense_kernel:
АА4
%assignvariableop_64_adam_m_dense_bias:	А4
%assignvariableop_65_adam_v_dense_bias:	А<
)assignvariableop_66_adam_m_dense_1_kernel:	А<
)assignvariableop_67_adam_v_dense_1_kernel:	А5
'assignvariableop_68_adam_m_dense_1_bias:5
'assignvariableop_69_adam_v_dense_1_bias:%
assignvariableop_70_total_1: %
assignvariableop_71_count_1: #
assignvariableop_72_total: #
assignvariableop_73_count: 
identity_75ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_8ҐAssignVariableOp_9Ђ 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*—
value«BƒKB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ш
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¬
_output_shapesѓ
ђ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_2_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_3_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_25AssignVariableOpassignvariableop_25_dense_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_1_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_1_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_28AssignVariableOpassignvariableop_28_iterationIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_29AssignVariableOp)assignvariableop_29_current_learning_rateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_m_conv3d_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_v_conv3d_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_m_conv3d_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_v_conv3d_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_m_batch_normalization_gammaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_v_batch_normalization_gammaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_m_batch_normalization_betaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_v_batch_normalization_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_conv3d_1_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_conv3d_1_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_conv3d_1_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_conv3d_1_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_m_batch_normalization_1_gammaIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_v_batch_normalization_1_gammaIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_m_batch_normalization_1_betaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adam_v_batch_normalization_1_betaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_conv3d_2_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_conv3d_2_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_m_conv3d_2_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_v_conv3d_2_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_m_batch_normalization_2_gammaIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_v_batch_normalization_2_gammaIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_m_batch_normalization_2_betaIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_v_batch_normalization_2_betaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_m_conv3d_3_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_v_conv3d_3_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_m_conv3d_3_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_v_conv3d_3_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_m_batch_normalization_3_gammaIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_v_batch_normalization_3_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_m_batch_normalization_3_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_v_batch_normalization_3_betaIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_m_dense_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_v_dense_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_m_dense_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_65AssignVariableOp%assignvariableop_65_adam_v_dense_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_m_dense_1_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_v_dense_1_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_m_dense_1_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_v_dense_1_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_70AssignVariableOpassignvariableop_70_total_1Identity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_71AssignVariableOpassignvariableop_71_count_1Identity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_72AssignVariableOpassignvariableop_72_totalIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_73AssignVariableOpassignvariableop_73_countIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Ђ
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_75IdentityIdentity_74:output:0^NoOp_1*
T0*
_output_shapes
: ф
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_75Identity_75:output:0*(
_construction_contextkEagerRuntime*Ђ
_input_shapesЩ
Ц: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%J!

_user_specified_namecount:%I!

_user_specified_nametotal:'H#
!
_user_specified_name	count_1:'G#
!
_user_specified_name	total_1:3F/
-
_user_specified_nameAdam/v/dense_1/bias:3E/
-
_user_specified_nameAdam/m/dense_1/bias:5D1
/
_user_specified_nameAdam/v/dense_1/kernel:5C1
/
_user_specified_nameAdam/m/dense_1/kernel:1B-
+
_user_specified_nameAdam/v/dense/bias:1A-
+
_user_specified_nameAdam/m/dense/bias:3@/
-
_user_specified_nameAdam/v/dense/kernel:3?/
-
_user_specified_nameAdam/m/dense/kernel:A>=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:A==
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:B<>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:B;>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:4:0
.
_user_specified_nameAdam/v/conv3d_3/bias:490
.
_user_specified_nameAdam/m/conv3d_3/bias:682
0
_user_specified_nameAdam/v/conv3d_3/kernel:672
0
_user_specified_nameAdam/m/conv3d_3/kernel:A6=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A5=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B4>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B3>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:420
.
_user_specified_nameAdam/v/conv3d_2/bias:410
.
_user_specified_nameAdam/m/conv3d_2/bias:602
0
_user_specified_nameAdam/v/conv3d_2/kernel:6/2
0
_user_specified_nameAdam/m/conv3d_2/kernel:A.=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A-=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B,>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B+>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:4*0
.
_user_specified_nameAdam/v/conv3d_1/bias:4)0
.
_user_specified_nameAdam/m/conv3d_1/bias:6(2
0
_user_specified_nameAdam/v/conv3d_1/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_1/kernel:?&;
9
_user_specified_name!Adam/v/batch_normalization/beta:?%;
9
_user_specified_name!Adam/m/batch_normalization/beta:@$<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@#<
:
_user_specified_name" Adam/m/batch_normalization/gamma:2".
,
_user_specified_nameAdam/v/conv3d/bias:2!.
,
_user_specified_nameAdam/m/conv3d/bias:4 0
.
_user_specified_nameAdam/v/conv3d/kernel:40
.
_user_specified_nameAdam/m/conv3d/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:-)
'
_user_specified_nameconv3d_3/bias:/+
)
_user_specified_nameconv3d_3/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:-)
'
_user_specified_nameconv3d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
≠
†
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278716

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
з
ƒ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277830

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ѓ
K
"__inference__update_step_xla_29325
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
я
Г
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278022

inputs=
conv3d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpБ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0†
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
Њ

ѕ
4__inference_batch_normalization_layer_call_fn_278483

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277686Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278479:&"
 
_user_specified_name278477:&"
 
_user_specified_name278475:&"
 
_user_specified_name278473:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
џ
Z
"__inference__update_step_xla_29275
gradient&
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*+
_input_shapes
:@@: *
	_noinline(:($
"
_user_specified_name
variable:T P
*
_output_shapes
:@@
"
_user_specified_name
gradient
Э
Ь
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277776

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ў
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_277735

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ы
Ъ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278532

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@М
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
г
Д
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278048

inputs>
conv3d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpВ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0†
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
ў
g
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_278562

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
г
Д
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278736

inputs>
conv3d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpВ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0†
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
”
p
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_278819

inputs
identityk
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
–
a
(__inference_dropout_layer_call_fn_278844

inputs
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_278092p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≠
†
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278808

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
„
ј
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278606

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
‘

х
A__inference_dense_layer_call_and_return_conditional_losses_278075

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29285
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
Ќ

х
C__inference_dense_1_layer_call_and_return_conditional_losses_278104

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Џ
a
C__inference_dropout_layer_call_and_return_conditional_losses_278866

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ј

ѕ
4__inference_batch_normalization_layer_call_fn_278496

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277704Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278492:&"
 
_user_specified_name278490:&"
 
_user_specified_name278488:&"
 
_user_specified_name278486:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
п
L
0__inference_max_pooling3d_1_layer_call_fn_278557

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_277735Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Щ
Д
&__inference_3dcnn_layer_call_fn_278253
input_1%
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@'
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@)

unknown_11:@А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А*

unknown_17:АА

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:
identityИҐStatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_3dcnn_layer_call_and_return_conditional_losses_278111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278249:&"
 
_user_specified_name278247:&"
 
_user_specified_name278245:&"
 
_user_specified_name278243:&"
 
_user_specified_name278241:&"
 
_user_specified_name278239:&"
 
_user_specified_name278237:&"
 
_user_specified_name278235:&"
 
_user_specified_name278233:&"
 
_user_specified_name278231:&"
 
_user_specified_name278229:&"
 
_user_specified_name278227:&"
 
_user_specified_name278225:&"
 
_user_specified_name278223:&"
 
_user_specified_name278221:&"
 
_user_specified_name278219:&"
 
_user_specified_name278217:&"
 
_user_specified_name278215:&
"
 
_user_specified_name278213:&	"
 
_user_specified_name278211:&"
 
_user_specified_name278209:&"
 
_user_specified_name278207:&"
 
_user_specified_name278205:&"
 
_user_specified_name278203:&"
 
_user_specified_name278201:&"
 
_user_specified_name278199:&"
 
_user_specified_name278197:&"
 
_user_specified_name278195:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
ƒ

—
6__inference_batch_normalization_1_layer_call_fn_278588

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall™
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277776Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278584:&"
 
_user_specified_name278582:&"
 
_user_specified_name278580:&"
 
_user_specified_name278578:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Ґ
D
(__inference_dropout_layer_call_fn_278849

inputs
identity≤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_278184a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ю

b
C__inference_dropout_layer_call_and_return_conditional_losses_278092

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::нѕН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ў
g
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_277879

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
„
Б
D__inference_conv3d_1_layer_call_and_return_conditional_losses_278552

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€==@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€==@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€??@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:€€€€€€€€€??@
 
_user_specified_nameinputs
хW
д
A__inference_3dcnn_layer_call_and_return_conditional_losses_278192
input_1+
conv3d_278114:@
conv3d_278116:@(
batch_normalization_278120:@(
batch_normalization_278122:@(
batch_normalization_278124:@(
batch_normalization_278126:@-
conv3d_1_278129:@@
conv3d_1_278131:@*
batch_normalization_1_278135:@*
batch_normalization_1_278137:@*
batch_normalization_1_278139:@*
batch_normalization_1_278141:@.
conv3d_2_278144:@А
conv3d_2_278146:	А+
batch_normalization_2_278150:	А+
batch_normalization_2_278152:	А+
batch_normalization_2_278154:	А+
batch_normalization_2_278156:	А/
conv3d_3_278159:АА
conv3d_3_278161:	А+
batch_normalization_3_278165:	А+
batch_normalization_3_278167:	А+
batch_normalization_3_278169:	А+
batch_normalization_3_278171:	А 
dense_278175:
АА
dense_278177:	А!
dense_1_278186:	А
dense_1_278188:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐconv3d/StatefulPartitionedCallҐ conv3d_1/StatefulPartitionedCallҐ conv3d_2/StatefulPartitionedCallҐ conv3d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallш
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_278114conv3d_278116*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€~~>@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv3d_layer_call_and_return_conditional_losses_277970т
max_pooling3d/PartitionedCallPartitionedCall'conv3d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€??@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_277663З
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0batch_normalization_278120batch_normalization_278122batch_normalization_278124batch_normalization_278126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€??@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_277704≠
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv3d_1_278129conv3d_1_278131*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€==@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_1_layer_call_and_return_conditional_losses_277996ш
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_277735Х
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0batch_normalization_1_278135batch_normalization_1_278137batch_normalization_1_278139batch_normalization_1_278141*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277776∞
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv3d_2_278144conv3d_2_278146*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278022щ
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_277807Ц
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0batch_normalization_2_278150batch_normalization_2_278152batch_normalization_2_278154batch_normalization_2_278156*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277848∞
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv3d_3_278159conv3d_3_278161*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278048щ
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_277879Ц
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_3/PartitionedCall:output:0batch_normalization_3_278165batch_normalization_3_278167batch_normalization_3_278169batch_normalization_3_278171*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277920М
(global_average_pooling3d/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_277952У
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling3d/PartitionedCall:output:0dense_278175dense_278177*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_278075Џ
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_278184Й
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_278186dense_1_278188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_278104w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ђ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:&"
 
_user_specified_name278188:&"
 
_user_specified_name278186:&"
 
_user_specified_name278177:&"
 
_user_specified_name278175:&"
 
_user_specified_name278171:&"
 
_user_specified_name278169:&"
 
_user_specified_name278167:&"
 
_user_specified_name278165:&"
 
_user_specified_name278161:&"
 
_user_specified_name278159:&"
 
_user_specified_name278156:&"
 
_user_specified_name278154:&"
 
_user_specified_name278152:&"
 
_user_specified_name278150:&"
 
_user_specified_name278146:&"
 
_user_specified_name278144:&"
 
_user_specified_name278141:&"
 
_user_specified_name278139:&
"
 
_user_specified_name278137:&	"
 
_user_specified_name278135:&"
 
_user_specified_name278131:&"
 
_user_specified_name278129:&"
 
_user_specified_name278126:&"
 
_user_specified_name278124:&"
 
_user_specified_name278122:&"
 
_user_specified_name278120:&"
 
_user_specified_name278116:&"
 
_user_specified_name278114:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
з
ƒ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278790

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
і
•
)__inference_conv3d_3_layer_call_fn_278725

inputs'
unknown:АА
	unknown_0:	А
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278048|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278721:&"
 
_user_specified_name278719:\ X
4
_output_shapes"
 :€€€€€€€€€А
 
_user_specified_nameinputs
Ї
O
"__inference__update_step_xla_29345
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	А: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	А
"
_user_specified_name
gradient
°
Д
&__inference_3dcnn_layer_call_fn_278314
input_1%
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@'
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@)

unknown_11:@А

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А*

unknown_17:АА

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23:
АА

unknown_24:	А

unknown_25:	А

unknown_26:
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_3dcnn_layer_call_and_return_conditional_losses_278192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278310:&"
 
_user_specified_name278308:&"
 
_user_specified_name278306:&"
 
_user_specified_name278304:&"
 
_user_specified_name278302:&"
 
_user_specified_name278300:&"
 
_user_specified_name278298:&"
 
_user_specified_name278296:&"
 
_user_specified_name278294:&"
 
_user_specified_name278292:&"
 
_user_specified_name278290:&"
 
_user_specified_name278288:&"
 
_user_specified_name278286:&"
 
_user_specified_name278284:&"
 
_user_specified_name278282:&"
 
_user_specified_name278280:&"
 
_user_specified_name278278:&"
 
_user_specified_name278276:&
"
 
_user_specified_name278274:&	"
 
_user_specified_name278272:&"
 
_user_specified_name278270:&"
 
_user_specified_name278268:&"
 
_user_specified_name278266:&"
 
_user_specified_name278264:&"
 
_user_specified_name278262:&"
 
_user_specified_name278260:&"
 
_user_specified_name278258:&"
 
_user_specified_name278256:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
цґ
—F
__inference__traced_save_279352
file_prefixB
$read_disablecopyonread_conv3d_kernel:@2
$read_1_disablecopyonread_conv3d_bias:@@
2read_2_disablecopyonread_batch_normalization_gamma:@?
1read_3_disablecopyonread_batch_normalization_beta:@F
8read_4_disablecopyonread_batch_normalization_moving_mean:@J
<read_5_disablecopyonread_batch_normalization_moving_variance:@F
(read_6_disablecopyonread_conv3d_1_kernel:@@4
&read_7_disablecopyonread_conv3d_1_bias:@B
4read_8_disablecopyonread_batch_normalization_1_gamma:@A
3read_9_disablecopyonread_batch_normalization_1_beta:@I
;read_10_disablecopyonread_batch_normalization_1_moving_mean:@M
?read_11_disablecopyonread_batch_normalization_1_moving_variance:@H
)read_12_disablecopyonread_conv3d_2_kernel:@А6
'read_13_disablecopyonread_conv3d_2_bias:	АD
5read_14_disablecopyonread_batch_normalization_2_gamma:	АC
4read_15_disablecopyonread_batch_normalization_2_beta:	АJ
;read_16_disablecopyonread_batch_normalization_2_moving_mean:	АN
?read_17_disablecopyonread_batch_normalization_2_moving_variance:	АI
)read_18_disablecopyonread_conv3d_3_kernel:АА6
'read_19_disablecopyonread_conv3d_3_bias:	АD
5read_20_disablecopyonread_batch_normalization_3_gamma:	АC
4read_21_disablecopyonread_batch_normalization_3_beta:	АJ
;read_22_disablecopyonread_batch_normalization_3_moving_mean:	АN
?read_23_disablecopyonread_batch_normalization_3_moving_variance:	А:
&read_24_disablecopyonread_dense_kernel:
АА3
$read_25_disablecopyonread_dense_bias:	А;
(read_26_disablecopyonread_dense_1_kernel:	А4
&read_27_disablecopyonread_dense_1_bias:-
#read_28_disablecopyonread_iteration:	 9
/read_29_disablecopyonread_current_learning_rate: L
.read_30_disablecopyonread_adam_m_conv3d_kernel:@L
.read_31_disablecopyonread_adam_v_conv3d_kernel:@:
,read_32_disablecopyonread_adam_m_conv3d_bias:@:
,read_33_disablecopyonread_adam_v_conv3d_bias:@H
:read_34_disablecopyonread_adam_m_batch_normalization_gamma:@H
:read_35_disablecopyonread_adam_v_batch_normalization_gamma:@G
9read_36_disablecopyonread_adam_m_batch_normalization_beta:@G
9read_37_disablecopyonread_adam_v_batch_normalization_beta:@N
0read_38_disablecopyonread_adam_m_conv3d_1_kernel:@@N
0read_39_disablecopyonread_adam_v_conv3d_1_kernel:@@<
.read_40_disablecopyonread_adam_m_conv3d_1_bias:@<
.read_41_disablecopyonread_adam_v_conv3d_1_bias:@J
<read_42_disablecopyonread_adam_m_batch_normalization_1_gamma:@J
<read_43_disablecopyonread_adam_v_batch_normalization_1_gamma:@I
;read_44_disablecopyonread_adam_m_batch_normalization_1_beta:@I
;read_45_disablecopyonread_adam_v_batch_normalization_1_beta:@O
0read_46_disablecopyonread_adam_m_conv3d_2_kernel:@АO
0read_47_disablecopyonread_adam_v_conv3d_2_kernel:@А=
.read_48_disablecopyonread_adam_m_conv3d_2_bias:	А=
.read_49_disablecopyonread_adam_v_conv3d_2_bias:	АK
<read_50_disablecopyonread_adam_m_batch_normalization_2_gamma:	АK
<read_51_disablecopyonread_adam_v_batch_normalization_2_gamma:	АJ
;read_52_disablecopyonread_adam_m_batch_normalization_2_beta:	АJ
;read_53_disablecopyonread_adam_v_batch_normalization_2_beta:	АP
0read_54_disablecopyonread_adam_m_conv3d_3_kernel:ААP
0read_55_disablecopyonread_adam_v_conv3d_3_kernel:АА=
.read_56_disablecopyonread_adam_m_conv3d_3_bias:	А=
.read_57_disablecopyonread_adam_v_conv3d_3_bias:	АK
<read_58_disablecopyonread_adam_m_batch_normalization_3_gamma:	АK
<read_59_disablecopyonread_adam_v_batch_normalization_3_gamma:	АJ
;read_60_disablecopyonread_adam_m_batch_normalization_3_beta:	АJ
;read_61_disablecopyonread_adam_v_batch_normalization_3_beta:	АA
-read_62_disablecopyonread_adam_m_dense_kernel:
ААA
-read_63_disablecopyonread_adam_v_dense_kernel:
АА:
+read_64_disablecopyonread_adam_m_dense_bias:	А:
+read_65_disablecopyonread_adam_v_dense_bias:	АB
/read_66_disablecopyonread_adam_m_dense_1_kernel:	АB
/read_67_disablecopyonread_adam_v_dense_1_kernel:	А;
-read_68_disablecopyonread_adam_m_dense_1_bias:;
-read_69_disablecopyonread_adam_v_dense_1_bias:+
!read_70_disablecopyonread_total_1: +
!read_71_disablecopyonread_count_1: )
read_72_disablecopyonread_total: )
read_73_disablecopyonread_count: 
savev2_const
identity_149ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_57/DisableCopyOnReadҐRead_57/ReadVariableOpҐRead_58/DisableCopyOnReadҐRead_58/ReadVariableOpҐRead_59/DisableCopyOnReadҐRead_59/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_60/DisableCopyOnReadҐRead_60/ReadVariableOpҐRead_61/DisableCopyOnReadҐRead_61/ReadVariableOpҐRead_62/DisableCopyOnReadҐRead_62/ReadVariableOpҐRead_63/DisableCopyOnReadҐRead_63/ReadVariableOpҐRead_64/DisableCopyOnReadҐRead_64/ReadVariableOpҐRead_65/DisableCopyOnReadҐRead_65/ReadVariableOpҐRead_66/DisableCopyOnReadҐRead_66/ReadVariableOpҐRead_67/DisableCopyOnReadҐRead_67/ReadVariableOpҐRead_68/DisableCopyOnReadҐRead_68/ReadVariableOpҐRead_69/DisableCopyOnReadҐRead_69/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_70/DisableCopyOnReadҐRead_70/ReadVariableOpҐRead_71/DisableCopyOnReadҐRead_71/ReadVariableOpҐRead_72/DisableCopyOnReadҐRead_72/ReadVariableOpҐRead_73/DisableCopyOnReadҐRead_73/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv3d_kernel"/device:CPU:0*
_output_shapes
 ђ
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv3d_kernel^Read/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0u
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@m

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0**
_output_shapes
:@x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv3d_bias"/device:CPU:0*
_output_shapes
 †
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv3d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ж
Read_2/DisableCopyOnReadDisableCopyOnRead2read_2_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Ѓ
Read_2/ReadVariableOpReadVariableOp2read_2_disablecopyonread_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 ≠
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@М
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 і
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_batch_normalization_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_5/DisableCopyOnReadDisableCopyOnRead<read_5_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 Є
Read_5/ReadVariableOpReadVariableOp<read_5_disablecopyonread_batch_normalization_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 і
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv3d_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@@*
dtype0z
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@@q
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0**
_output_shapes
:@@z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv3d_1_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv3d_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@И
Read_8/DisableCopyOnReadDisableCopyOnRead4read_8_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 ∞
Read_8/ReadVariableOpReadVariableOp4read_8_disablecopyonread_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_9/DisableCopyOnReadDisableCopyOnRead3read_9_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 ѓ
Read_9/ReadVariableOpReadVariableOp3read_9_disablecopyonread_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_10/DisableCopyOnReadDisableCopyOnRead;read_10_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 є
Read_10/ReadVariableOpReadVariableOp;read_10_disablecopyonread_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ф
Read_11/DisableCopyOnReadDisableCopyOnRead?read_11_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 љ
Read_11/ReadVariableOpReadVariableOp?read_11_disablecopyonread_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 Є
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_conv3d_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0|
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аr
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*+
_output_shapes
:@А|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_conv3d_2_bias"/device:CPU:0*
_output_shapes
 ¶
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_conv3d_2_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:АК
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 і
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_2_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЙ
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 ≥
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_2_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_2_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:АФ
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_2_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:А~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 є
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_conv3d_3_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0}
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААs
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*,
_output_shapes
:АА|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_conv3d_3_bias"/device:CPU:0*
_output_shapes
 ¶
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_conv3d_3_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:АК
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 і
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_batch_normalization_3_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЙ
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 ≥
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_batch_normalization_3_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_22/DisableCopyOnReadDisableCopyOnRead;read_22_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 Ї
Read_22/ReadVariableOpReadVariableOp;read_22_disablecopyonread_batch_normalization_3_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:АФ
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 Њ
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_batch_normalization_3_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:А{
Read_24/DisableCopyOnReadDisableCopyOnRead&read_24_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 ™
Read_24/ReadVariableOpReadVariableOp&read_24_disablecopyonread_dense_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААy
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 £
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_dense_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_26/DisableCopyOnReadDisableCopyOnRead(read_26_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_26/ReadVariableOpReadVariableOp(read_26_disablecopyonread_dense_1_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	А{
Read_27/DisableCopyOnReadDisableCopyOnRead&read_27_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 §
Read_27/ReadVariableOpReadVariableOp&read_27_disablecopyonread_dense_1_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_28/DisableCopyOnReadDisableCopyOnRead#read_28_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_28/ReadVariableOpReadVariableOp#read_28_disablecopyonread_iteration^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0	*
_output_shapes
: Д
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_current_learning_rate"/device:CPU:0*
_output_shapes
 ©
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_current_learning_rate^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_m_conv3d_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_m_conv3d_kernel^Read_30/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0{
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@q
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0**
_output_shapes
:@Г
Read_31/DisableCopyOnReadDisableCopyOnRead.read_31_disablecopyonread_adam_v_conv3d_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_31/ReadVariableOpReadVariableOp.read_31_disablecopyonread_adam_v_conv3d_kernel^Read_31/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@*
dtype0{
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@q
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0**
_output_shapes
:@Б
Read_32/DisableCopyOnReadDisableCopyOnRead,read_32_disablecopyonread_adam_m_conv3d_bias"/device:CPU:0*
_output_shapes
 ™
Read_32/ReadVariableOpReadVariableOp,read_32_disablecopyonread_adam_m_conv3d_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@Б
Read_33/DisableCopyOnReadDisableCopyOnRead,read_33_disablecopyonread_adam_v_conv3d_bias"/device:CPU:0*
_output_shapes
 ™
Read_33/ReadVariableOpReadVariableOp,read_33_disablecopyonread_adam_v_conv3d_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_34/DisableCopyOnReadDisableCopyOnRead:read_34_disablecopyonread_adam_m_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Є
Read_34/ReadVariableOpReadVariableOp:read_34_disablecopyonread_adam_m_batch_normalization_gamma^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@П
Read_35/DisableCopyOnReadDisableCopyOnRead:read_35_disablecopyonread_adam_v_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 Є
Read_35/ReadVariableOpReadVariableOp:read_35_disablecopyonread_adam_v_batch_normalization_gamma^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@О
Read_36/DisableCopyOnReadDisableCopyOnRead9read_36_disablecopyonread_adam_m_batch_normalization_beta"/device:CPU:0*
_output_shapes
 Ј
Read_36/ReadVariableOpReadVariableOp9read_36_disablecopyonread_adam_m_batch_normalization_beta^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@О
Read_37/DisableCopyOnReadDisableCopyOnRead9read_37_disablecopyonread_adam_v_batch_normalization_beta"/device:CPU:0*
_output_shapes
 Ј
Read_37/ReadVariableOpReadVariableOp9read_37_disablecopyonread_adam_v_batch_normalization_beta^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_conv3d_1_kernel^Read_38/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@@*
dtype0{
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@@q
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0**
_output_shapes
:@@Е
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_conv3d_1_kernel"/device:CPU:0*
_output_shapes
 Њ
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_conv3d_1_kernel^Read_39/DisableCopyOnRead"/device:CPU:0**
_output_shapes
:@@*
dtype0{
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0**
_output_shapes
:@@q
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0**
_output_shapes
:@@Г
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_conv3d_1_bias"/device:CPU:0*
_output_shapes
 ђ
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_conv3d_1_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@Г
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_conv3d_1_bias"/device:CPU:0*
_output_shapes
 ђ
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_conv3d_1_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_42/DisableCopyOnReadDisableCopyOnRead<read_42_disablecopyonread_adam_m_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_42/ReadVariableOpReadVariableOp<read_42_disablecopyonread_adam_m_batch_normalization_1_gamma^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:@С
Read_43/DisableCopyOnReadDisableCopyOnRead<read_43_disablecopyonread_adam_v_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 Ї
Read_43/ReadVariableOpReadVariableOp<read_43_disablecopyonread_adam_v_batch_normalization_1_gamma^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_44/DisableCopyOnReadDisableCopyOnRead;read_44_disablecopyonread_adam_m_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 є
Read_44/ReadVariableOpReadVariableOp;read_44_disablecopyonread_adam_m_batch_normalization_1_beta^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_45/DisableCopyOnReadDisableCopyOnRead;read_45_disablecopyonread_adam_v_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 є
Read_45/ReadVariableOpReadVariableOp;read_45_disablecopyonread_adam_v_batch_normalization_1_beta^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 њ
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_conv3d_2_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0|
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аr
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*+
_output_shapes
:@АЕ
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_conv3d_2_kernel"/device:CPU:0*
_output_shapes
 њ
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_conv3d_2_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*+
_output_shapes
:@А*
dtype0|
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*+
_output_shapes
:@Аr
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*+
_output_shapes
:@АГ
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_m_conv3d_2_bias"/device:CPU:0*
_output_shapes
 ≠
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_m_conv3d_2_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_49/DisableCopyOnReadDisableCopyOnRead.read_49_disablecopyonread_adam_v_conv3d_2_bias"/device:CPU:0*
_output_shapes
 ≠
Read_49/ReadVariableOpReadVariableOp.read_49_disablecopyonread_adam_v_conv3d_2_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_50/DisableCopyOnReadDisableCopyOnRead<read_50_disablecopyonread_adam_m_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 ї
Read_50/ReadVariableOpReadVariableOp<read_50_disablecopyonread_adam_m_batch_normalization_2_gamma^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_51/DisableCopyOnReadDisableCopyOnRead<read_51_disablecopyonread_adam_v_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 ї
Read_51/ReadVariableOpReadVariableOp<read_51_disablecopyonread_adam_v_batch_normalization_2_gamma^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_52/DisableCopyOnReadDisableCopyOnRead;read_52_disablecopyonread_adam_m_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 Ї
Read_52/ReadVariableOpReadVariableOp;read_52_disablecopyonread_adam_m_batch_normalization_2_beta^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_53/DisableCopyOnReadDisableCopyOnRead;read_53_disablecopyonread_adam_v_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 Ї
Read_53/ReadVariableOpReadVariableOp;read_53_disablecopyonread_adam_v_batch_normalization_2_beta^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_m_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 ј
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_m_conv3d_3_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0~
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААu
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*,
_output_shapes
:ААЕ
Read_55/DisableCopyOnReadDisableCopyOnRead0read_55_disablecopyonread_adam_v_conv3d_3_kernel"/device:CPU:0*
_output_shapes
 ј
Read_55/ReadVariableOpReadVariableOp0read_55_disablecopyonread_adam_v_conv3d_3_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*,
_output_shapes
:АА*
dtype0~
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*,
_output_shapes
:ААu
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*,
_output_shapes
:ААГ
Read_56/DisableCopyOnReadDisableCopyOnRead.read_56_disablecopyonread_adam_m_conv3d_3_bias"/device:CPU:0*
_output_shapes
 ≠
Read_56/ReadVariableOpReadVariableOp.read_56_disablecopyonread_adam_m_conv3d_3_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:АГ
Read_57/DisableCopyOnReadDisableCopyOnRead.read_57_disablecopyonread_adam_v_conv3d_3_bias"/device:CPU:0*
_output_shapes
 ≠
Read_57/ReadVariableOpReadVariableOp.read_57_disablecopyonread_adam_v_conv3d_3_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_58/DisableCopyOnReadDisableCopyOnRead<read_58_disablecopyonread_adam_m_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 ї
Read_58/ReadVariableOpReadVariableOp<read_58_disablecopyonread_adam_m_batch_normalization_3_gamma^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:АС
Read_59/DisableCopyOnReadDisableCopyOnRead<read_59_disablecopyonread_adam_v_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 ї
Read_59/ReadVariableOpReadVariableOp<read_59_disablecopyonread_adam_v_batch_normalization_3_gamma^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_60/DisableCopyOnReadDisableCopyOnRead;read_60_disablecopyonread_adam_m_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Ї
Read_60/ReadVariableOpReadVariableOp;read_60_disablecopyonread_adam_m_batch_normalization_3_beta^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:АР
Read_61/DisableCopyOnReadDisableCopyOnRead;read_61_disablecopyonread_adam_v_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 Ї
Read_61/ReadVariableOpReadVariableOp;read_61_disablecopyonread_adam_v_batch_normalization_3_beta^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_62/DisableCopyOnReadDisableCopyOnRead-read_62_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 ±
Read_62/ReadVariableOpReadVariableOp-read_62_disablecopyonread_adam_m_dense_kernel^Read_62/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_63/DisableCopyOnReadDisableCopyOnRead-read_63_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 ±
Read_63/ReadVariableOpReadVariableOp-read_63_disablecopyonread_adam_v_dense_kernel^Read_63/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0r
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААi
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААА
Read_64/DisableCopyOnReadDisableCopyOnRead+read_64_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 ™
Read_64/ReadVariableOpReadVariableOp+read_64_disablecopyonread_adam_m_dense_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:АА
Read_65/DisableCopyOnReadDisableCopyOnRead+read_65_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 ™
Read_65/ReadVariableOpReadVariableOp+read_65_disablecopyonread_adam_v_dense_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_66/DisableCopyOnReadDisableCopyOnRead/read_66_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_66/ReadVariableOpReadVariableOp/read_66_disablecopyonread_adam_m_dense_1_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:	АД
Read_67/DisableCopyOnReadDisableCopyOnRead/read_67_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_67/ReadVariableOpReadVariableOp/read_67_disablecopyonread_adam_v_dense_1_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0q
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аh
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_68/DisableCopyOnReadDisableCopyOnRead-read_68_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_68/ReadVariableOpReadVariableOp-read_68_disablecopyonread_adam_m_dense_1_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_69/DisableCopyOnReadDisableCopyOnRead-read_69_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_69/ReadVariableOpReadVariableOp-read_69_disablecopyonread_adam_v_dense_1_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_70/DisableCopyOnReadDisableCopyOnRead!read_70_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_70/ReadVariableOpReadVariableOp!read_70_disablecopyonread_total_1^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_71/DisableCopyOnReadDisableCopyOnRead!read_71_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_71/ReadVariableOpReadVariableOp!read_71_disablecopyonread_count_1^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_72/DisableCopyOnReadDisableCopyOnReadread_72_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_72/ReadVariableOpReadVariableOpread_72_disablecopyonread_total^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_73/DisableCopyOnReadDisableCopyOnReadread_73_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_73/ReadVariableOpReadVariableOpread_73_disablecopyonread_count^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
: ® 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*—
value«BƒKB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B С
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Y
dtypesO
M2K	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_148Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_149IdentityIdentity_148:output:0^NoOp*
T0*
_output_shapes
: с
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_149Identity_149:output:0*(
_construction_contextkEagerRuntime*≠
_input_shapesЫ
Ш: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=K9

_output_shapes
: 

_user_specified_nameConst:%J!

_user_specified_namecount:%I!

_user_specified_nametotal:'H#
!
_user_specified_name	count_1:'G#
!
_user_specified_name	total_1:3F/
-
_user_specified_nameAdam/v/dense_1/bias:3E/
-
_user_specified_nameAdam/m/dense_1/bias:5D1
/
_user_specified_nameAdam/v/dense_1/kernel:5C1
/
_user_specified_nameAdam/m/dense_1/kernel:1B-
+
_user_specified_nameAdam/v/dense/bias:1A-
+
_user_specified_nameAdam/m/dense/bias:3@/
-
_user_specified_nameAdam/v/dense/kernel:3?/
-
_user_specified_nameAdam/m/dense/kernel:A>=
;
_user_specified_name#!Adam/v/batch_normalization_3/beta:A==
;
_user_specified_name#!Adam/m/batch_normalization_3/beta:B<>
<
_user_specified_name$"Adam/v/batch_normalization_3/gamma:B;>
<
_user_specified_name$"Adam/m/batch_normalization_3/gamma:4:0
.
_user_specified_nameAdam/v/conv3d_3/bias:490
.
_user_specified_nameAdam/m/conv3d_3/bias:682
0
_user_specified_nameAdam/v/conv3d_3/kernel:672
0
_user_specified_nameAdam/m/conv3d_3/kernel:A6=
;
_user_specified_name#!Adam/v/batch_normalization_2/beta:A5=
;
_user_specified_name#!Adam/m/batch_normalization_2/beta:B4>
<
_user_specified_name$"Adam/v/batch_normalization_2/gamma:B3>
<
_user_specified_name$"Adam/m/batch_normalization_2/gamma:420
.
_user_specified_nameAdam/v/conv3d_2/bias:410
.
_user_specified_nameAdam/m/conv3d_2/bias:602
0
_user_specified_nameAdam/v/conv3d_2/kernel:6/2
0
_user_specified_nameAdam/m/conv3d_2/kernel:A.=
;
_user_specified_name#!Adam/v/batch_normalization_1/beta:A-=
;
_user_specified_name#!Adam/m/batch_normalization_1/beta:B,>
<
_user_specified_name$"Adam/v/batch_normalization_1/gamma:B+>
<
_user_specified_name$"Adam/m/batch_normalization_1/gamma:4*0
.
_user_specified_nameAdam/v/conv3d_1/bias:4)0
.
_user_specified_nameAdam/m/conv3d_1/bias:6(2
0
_user_specified_nameAdam/v/conv3d_1/kernel:6'2
0
_user_specified_nameAdam/m/conv3d_1/kernel:?&;
9
_user_specified_name!Adam/v/batch_normalization/beta:?%;
9
_user_specified_name!Adam/m/batch_normalization/beta:@$<
:
_user_specified_name" Adam/v/batch_normalization/gamma:@#<
:
_user_specified_name" Adam/m/batch_normalization/gamma:2".
,
_user_specified_nameAdam/v/conv3d/bias:2!.
,
_user_specified_nameAdam/m/conv3d/bias:4 0
.
_user_specified_nameAdam/v/conv3d/kernel:40
.
_user_specified_nameAdam/m/conv3d/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:-)
'
_user_specified_nameconv3d_3/bias:/+
)
_user_specified_nameconv3d_3/kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:-)
'
_user_specified_nameconv3d_2/bias:/+
)
_user_specified_nameconv3d_2/kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::
6
4
_user_specified_namebatch_normalization_1/beta:;	7
5
_user_specified_namebatch_normalization_1/gamma:-)
'
_user_specified_nameconv3d_1/bias:/+
)
_user_specified_nameconv3d_1/kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?;
9
_user_specified_name!batch_normalization/moving_mean:84
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:+'
%
_user_specified_nameconv3d/bias:-)
'
_user_specified_nameconv3d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ѓ
K
"__inference__update_step_xla_29310
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
¬

—
6__inference_batch_normalization_1_layer_call_fn_278575

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_277758Ц
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278571:&"
 
_user_specified_name278569:&"
 
_user_specified_name278567:&"
 
_user_specified_name278565:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ў
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_278654

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
п
L
0__inference_max_pooling3d_2_layer_call_fn_278649

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_277807Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѓ
K
"__inference__update_step_xla_29330
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
≠
†
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277920

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0т
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АМ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29350
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
±
§
)__inference_conv3d_2_layer_call_fn_278633

inputs&
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278022|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278629:&"
 
_user_specified_name278627:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
’
Њ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278514

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*j
_output_shapesX
V:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:v r
N
_output_shapes<
::8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
љ
P
"__inference__update_step_xla_29335
gradient
variable:
АА*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
АА: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:
АА
"
_user_specified_name
gradient
ў
g
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_277807

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
з
ƒ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_277902

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0А
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*o
_output_shapes]
[:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
exponential_avg_factor%
„#<∆
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(–
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Л
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
„
e
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_278470

inputs
identityЊ
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
*
paddingVALID*
strides	
К
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ђ
J
"__inference__update_step_xla_29270
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
я
Г
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278644

inputs=
conv3d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpБ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0†
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0В
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аn
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€АS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:€€€€€€€€€@
 
_user_specified_nameinputs
ў
€
B__inference_conv3d_layer_call_and_return_conditional_losses_277970

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€~~>@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:] Y
5
_output_shapes#
!:€€€€€€€€€АА@
 
_user_specified_nameinputs
„
Б
D__inference_conv3d_1_layer_call_and_return_conditional_losses_277996

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv3D/ReadVariableOpА
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0Я
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@*
paddingVALID*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Б
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€==@m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:€€€€€€€€€==@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€??@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:[ W
3
_output_shapes!
:€€€€€€€€€??@
 
_user_specified_nameinputs
Ѓ
K
"__inference__update_step_xla_29305
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
≤
U
9__inference_global_average_pooling3d_layer_call_fn_278813

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_277952i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ё
[
"__inference__update_step_xla_29295
gradient'
variable:@А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*,
_input_shapes
:@А: *
	_noinline(:($
"
_user_specified_name
variable:U Q
+
_output_shapes
:@А
"
_user_specified_name
gradient
Ѓ
K
"__inference__update_step_xla_29340
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:А
"
_user_specified_name
gradient
л
J
.__inference_max_pooling3d_layer_call_fn_278465

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_277663Р
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€: {
W
_output_shapesE
C:A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¶Ы
¬
!__inference__wrapped_model_277658
input_1H
*dcnn_conv3d_conv3d_readvariableop_resource:@9
+dcnn_conv3d_biasadd_readvariableop_resource:@>
0dcnn_batch_normalization_readvariableop_resource:@@
2dcnn_batch_normalization_readvariableop_1_resource:@O
Adcnn_batch_normalization_fusedbatchnormv3_readvariableop_resource:@Q
Cdcnn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@J
,dcnn_conv3d_1_conv3d_readvariableop_resource:@@;
-dcnn_conv3d_1_biasadd_readvariableop_resource:@@
2dcnn_batch_normalization_1_readvariableop_resource:@B
4dcnn_batch_normalization_1_readvariableop_1_resource:@Q
Cdcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@S
Edcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@K
,dcnn_conv3d_2_conv3d_readvariableop_resource:@А<
-dcnn_conv3d_2_biasadd_readvariableop_resource:	АA
2dcnn_batch_normalization_2_readvariableop_resource:	АC
4dcnn_batch_normalization_2_readvariableop_1_resource:	АR
Cdcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	АT
Edcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	АL
,dcnn_conv3d_3_conv3d_readvariableop_resource:АА<
-dcnn_conv3d_3_biasadd_readvariableop_resource:	АA
2dcnn_batch_normalization_3_readvariableop_resource:	АC
4dcnn_batch_normalization_3_readvariableop_1_resource:	АR
Cdcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АT
Edcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А=
)dcnn_dense_matmul_readvariableop_resource:
АА9
*dcnn_dense_biasadd_readvariableop_resource:	А>
+dcnn_dense_1_matmul_readvariableop_resource:	А:
,dcnn_dense_1_biasadd_readvariableop_resource:
identityИҐ93dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOpҐ;3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ(3dcnn/batch_normalization/ReadVariableOpҐ*3dcnn/batch_normalization/ReadVariableOp_1Ґ;3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ=3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ*3dcnn/batch_normalization_1/ReadVariableOpҐ,3dcnn/batch_normalization_1/ReadVariableOp_1Ґ;3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ=3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ*3dcnn/batch_normalization_2/ReadVariableOpҐ,3dcnn/batch_normalization_2/ReadVariableOp_1Ґ;3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ=3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ*3dcnn/batch_normalization_3/ReadVariableOpҐ,3dcnn/batch_normalization_3/ReadVariableOp_1Ґ#3dcnn/conv3d/BiasAdd/ReadVariableOpҐ"3dcnn/conv3d/Conv3D/ReadVariableOpҐ%3dcnn/conv3d_1/BiasAdd/ReadVariableOpҐ$3dcnn/conv3d_1/Conv3D/ReadVariableOpҐ%3dcnn/conv3d_2/BiasAdd/ReadVariableOpҐ$3dcnn/conv3d_2/Conv3D/ReadVariableOpҐ%3dcnn/conv3d_3/BiasAdd/ReadVariableOpҐ$3dcnn/conv3d_3/Conv3D/ReadVariableOpҐ"3dcnn/dense/BiasAdd/ReadVariableOpҐ!3dcnn/dense/MatMul/ReadVariableOpҐ$3dcnn/dense_1/BiasAdd/ReadVariableOpҐ#3dcnn/dense_1/MatMul/ReadVariableOpЩ
"3dcnn/conv3d/Conv3D/ReadVariableOpReadVariableOp*dcnn_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0Ї
3dcnn/conv3d/Conv3DConv3Dinput_1*3dcnn/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@*
paddingVALID*
strides	
Л
#3dcnn/conv3d/BiasAdd/ReadVariableOpReadVariableOp+dcnn_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0®
3dcnn/conv3d/BiasAddBiasAdd3dcnn/conv3d/Conv3D:output:0+3dcnn/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@v
3dcnn/conv3d/ReluRelu3dcnn/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€~~>@«
3dcnn/max_pooling3d/MaxPool3D	MaxPool3D3dcnn/conv3d/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€??@*
ksize	
*
paddingVALID*
strides	
Х
(3dcnn/batch_normalization/ReadVariableOpReadVariableOp0dcnn_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0Щ
*3dcnn/batch_normalization/ReadVariableOp_1ReadVariableOp2dcnn_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ј
93dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpAdcnn_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ї
;3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCdcnn_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ф
*3dcnn/batch_normalization/FusedBatchNormV3FusedBatchNormV3&3dcnn/max_pooling3d/MaxPool3D:output:003dcnn/batch_normalization/ReadVariableOp:value:023dcnn/batch_normalization/ReadVariableOp_1:value:0A3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0C3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:€€€€€€€€€??@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Э
$3dcnn/conv3d_1/Conv3D/ReadVariableOpReadVariableOp,dcnn_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0е
3dcnn/conv3d_1/Conv3DConv3D.3dcnn/batch_normalization/FusedBatchNormV3:y:0,3dcnn/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@*
paddingVALID*
strides	
П
%3dcnn/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ѓ
3dcnn/conv3d_1/BiasAddBiasAdd3dcnn/conv3d_1/Conv3D:output:0-3dcnn/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:€€€€€€€€€==@z
3dcnn/conv3d_1/ReluRelu3dcnn/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:€€€€€€€€€==@Ћ
3dcnn/max_pooling3d_1/MaxPool3D	MaxPool3D!3dcnn/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:€€€€€€€€€@*
ksize	
*
paddingVALID*
strides	
Щ
*3dcnn/batch_normalization_1/ReadVariableOpReadVariableOp2dcnn_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0Э
,3dcnn/batch_normalization_1/ReadVariableOp_1ReadVariableOp4dcnn_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0ї
;3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpCdcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0њ
=3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEdcnn_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0А
,3dcnn/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3(3dcnn/max_pooling3d_1/MaxPool3D:output:023dcnn/batch_normalization_1/ReadVariableOp:value:043dcnn/batch_normalization_1/ReadVariableOp_1:value:0C3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0E3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*O
_output_shapes=
;:€€€€€€€€€@:@:@:@:@:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Ю
$3dcnn/conv3d_2/Conv3D/ReadVariableOpReadVariableOp,dcnn_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@А*
dtype0и
3dcnn/conv3d_2/Conv3DConv3D03dcnn/batch_normalization_1/FusedBatchNormV3:y:0,3dcnn/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
Р
%3dcnn/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ѓ
3dcnn/conv3d_2/BiasAddBiasAdd3dcnn/conv3d_2/Conv3D:output:0-3dcnn/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А{
3dcnn/conv3d_2/ReluRelu3dcnn/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аћ
3dcnn/max_pooling3d_2/MaxPool3D	MaxPool3D!3dcnn/conv3d_2/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
Ъ
*3dcnn/batch_normalization_2/ReadVariableOpReadVariableOp2dcnn_batch_normalization_2_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
,3dcnn/batch_normalization_2/ReadVariableOp_1ReadVariableOp4dcnn_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Љ
;3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpCdcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
=3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEdcnn_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
,3dcnn/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3(3dcnn/max_pooling3d_2/MaxPool3D:output:023dcnn/batch_normalization_2/ReadVariableOp:value:043dcnn/batch_normalization_2/ReadVariableOp_1:value:0C3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0E3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( Я
$3dcnn/conv3d_3/Conv3D/ReadVariableOpReadVariableOp,dcnn_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:АА*
dtype0и
3dcnn/conv3d_3/Conv3DConv3D03dcnn/batch_normalization_2/FusedBatchNormV3:y:0,3dcnn/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
paddingVALID*
strides	
Р
%3dcnn/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp-dcnn_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ѓ
3dcnn/conv3d_3/BiasAddBiasAdd3dcnn/conv3d_3/Conv3D:output:0-3dcnn/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€А{
3dcnn/conv3d_3/ReluRelu3dcnn/conv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€Аћ
3dcnn/max_pooling3d_3/MaxPool3D	MaxPool3D!3dcnn/conv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :€€€€€€€€€А*
ksize	
*
paddingVALID*
strides	
Ъ
*3dcnn/batch_normalization_3/ReadVariableOpReadVariableOp2dcnn_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
,3dcnn/batch_normalization_3/ReadVariableOp_1ReadVariableOp4dcnn_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Љ
;3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpCdcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0ј
=3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEdcnn_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
,3dcnn/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3(3dcnn/max_pooling3d_3/MaxPool3D:output:023dcnn/batch_normalization_3/ReadVariableOp:value:043dcnn/batch_normalization_3/ReadVariableOp_1:value:0C3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0E3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*T
_output_shapesB
@:€€€€€€€€€А:А:А:А:А:*
data_formatNDHWC*
epsilon%oГ:*
is_training( К
53dcnn/global_average_pooling3d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         –
#3dcnn/global_average_pooling3d/MeanMean03dcnn/batch_normalization_3/FusedBatchNormV3:y:0>3dcnn/global_average_pooling3d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€АН
!3dcnn/dense/MatMul/ReadVariableOpReadVariableOp)dcnn_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0®
3dcnn/dense/MatMulMatMul,3dcnn/global_average_pooling3d/Mean:output:0)3dcnn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АК
"3dcnn/dense/BiasAdd/ReadVariableOpReadVariableOp*dcnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ы
3dcnn/dense/BiasAddBiasAdd3dcnn/dense/MatMul:product:0*3dcnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аi
3dcnn/dense/ReluRelu3dcnn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аu
3dcnn/dropout/IdentityIdentity3dcnn/dense/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АР
#3dcnn/dense_1/MatMul/ReadVariableOpReadVariableOp+dcnn_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ю
3dcnn/dense_1/MatMulMatMul3dcnn/dropout/Identity:output:0+3dcnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Н
$3dcnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp,dcnn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0†
3dcnn/dense_1/BiasAddBiasAdd3dcnn/dense_1/MatMul:product:0,3dcnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
3dcnn/dense_1/SigmoidSigmoid3dcnn/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentity3dcnn/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€–

NoOpNoOp:^3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp<^3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^3dcnn/batch_normalization/ReadVariableOp+^3dcnn/batch_normalization/ReadVariableOp_1<^3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^3dcnn/batch_normalization_1/ReadVariableOp-^3dcnn/batch_normalization_1/ReadVariableOp_1<^3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp>^3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1+^3dcnn/batch_normalization_2/ReadVariableOp-^3dcnn/batch_normalization_2/ReadVariableOp_1<^3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp>^3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1+^3dcnn/batch_normalization_3/ReadVariableOp-^3dcnn/batch_normalization_3/ReadVariableOp_1$^3dcnn/conv3d/BiasAdd/ReadVariableOp#^3dcnn/conv3d/Conv3D/ReadVariableOp&^3dcnn/conv3d_1/BiasAdd/ReadVariableOp%^3dcnn/conv3d_1/Conv3D/ReadVariableOp&^3dcnn/conv3d_2/BiasAdd/ReadVariableOp%^3dcnn/conv3d_2/Conv3D/ReadVariableOp&^3dcnn/conv3d_3/BiasAdd/ReadVariableOp%^3dcnn/conv3d_3/Conv3D/ReadVariableOp#^3dcnn/dense/BiasAdd/ReadVariableOp"^3dcnn/dense/MatMul/ReadVariableOp%^3dcnn/dense_1/BiasAdd/ReadVariableOp$^3dcnn/dense_1/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:€€€€€€€€€АА@: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2z
;3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;3dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp_12v
93dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp93dcnn/batch_normalization/FusedBatchNormV3/ReadVariableOp2X
*3dcnn/batch_normalization/ReadVariableOp_1*3dcnn/batch_normalization/ReadVariableOp_12T
(3dcnn/batch_normalization/ReadVariableOp(3dcnn/batch_normalization/ReadVariableOp2~
=3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12z
;3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;3dcnn/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2\
,3dcnn/batch_normalization_1/ReadVariableOp_1,3dcnn/batch_normalization_1/ReadVariableOp_12X
*3dcnn/batch_normalization_1/ReadVariableOp*3dcnn/batch_normalization_1/ReadVariableOp2~
=3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1=3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12z
;3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp;3dcnn/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2\
,3dcnn/batch_normalization_2/ReadVariableOp_1,3dcnn/batch_normalization_2/ReadVariableOp_12X
*3dcnn/batch_normalization_2/ReadVariableOp*3dcnn/batch_normalization_2/ReadVariableOp2~
=3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1=3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12z
;3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp;3dcnn/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2\
,3dcnn/batch_normalization_3/ReadVariableOp_1,3dcnn/batch_normalization_3/ReadVariableOp_12X
*3dcnn/batch_normalization_3/ReadVariableOp*3dcnn/batch_normalization_3/ReadVariableOp2J
#3dcnn/conv3d/BiasAdd/ReadVariableOp#3dcnn/conv3d/BiasAdd/ReadVariableOp2H
"3dcnn/conv3d/Conv3D/ReadVariableOp"3dcnn/conv3d/Conv3D/ReadVariableOp2N
%3dcnn/conv3d_1/BiasAdd/ReadVariableOp%3dcnn/conv3d_1/BiasAdd/ReadVariableOp2L
$3dcnn/conv3d_1/Conv3D/ReadVariableOp$3dcnn/conv3d_1/Conv3D/ReadVariableOp2N
%3dcnn/conv3d_2/BiasAdd/ReadVariableOp%3dcnn/conv3d_2/BiasAdd/ReadVariableOp2L
$3dcnn/conv3d_2/Conv3D/ReadVariableOp$3dcnn/conv3d_2/Conv3D/ReadVariableOp2N
%3dcnn/conv3d_3/BiasAdd/ReadVariableOp%3dcnn/conv3d_3/BiasAdd/ReadVariableOp2L
$3dcnn/conv3d_3/Conv3D/ReadVariableOp$3dcnn/conv3d_3/Conv3D/ReadVariableOp2H
"3dcnn/dense/BiasAdd/ReadVariableOp"3dcnn/dense/BiasAdd/ReadVariableOp2F
!3dcnn/dense/MatMul/ReadVariableOp!3dcnn/dense/MatMul/ReadVariableOp2L
$3dcnn/dense_1/BiasAdd/ReadVariableOp$3dcnn/dense_1/BiasAdd/ReadVariableOp2J
#3dcnn/dense_1/MatMul/ReadVariableOp#3dcnn/dense_1/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
5
_output_shapes#
!:€€€€€€€€€АА@
!
_user_specified_name	input_1
т
Ц
&__inference_dense_layer_call_fn_278828

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_278075p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278824:&"
 
_user_specified_name278822:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ќ

х
C__inference_dense_1_layer_call_and_return_conditional_losses_278886

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
 

’
6__inference_batch_normalization_2_layer_call_fn_278667

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277830Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278663:&"
 
_user_specified_name278661:&"
 
_user_specified_name278659:&"
 
_user_specified_name278657:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
т
Ц
(__inference_dense_1_layer_call_fn_278875

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_278104o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278871:&"
 
_user_specified_name278869:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ћ

’
6__inference_batch_normalization_2_layer_call_fn_278680

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_277848Ч
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name278676:&"
 
_user_specified_name278674:&"
 
_user_specified_name278672:&"
 
_user_specified_name278670:w s
O
_output_shapes=
;:9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs"ІL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Є
serving_default§
I
input_1>
serving_default_input_1:0€€€€€€€€€АА@;
dense_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:≠в
Ћ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
 #_jit_compiled_convolution_op"
_tf_keras_layer
•
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
к
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0axis
	1gamma
2beta
3moving_mean
4moving_variance"
_tf_keras_layer
Ё
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias
 =_jit_compiled_convolution_op"
_tf_keras_layer
•
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
к
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
Ё
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
•
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
к
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
daxis
	egamma
fbeta
gmoving_mean
hmoving_variance"
_tf_keras_layer
Ё
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
 q_jit_compiled_convolution_op"
_tf_keras_layer
•
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
н
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses
~axis
	gamma
	Аbeta
Бmoving_mean
Вmoving_variance"
_tf_keras_layer
Ђ
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
√
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
Пkernel
	Рbias"
_tf_keras_layer
√
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses
Ч_random_generator"
_tf_keras_layer
√
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ы	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses
Юkernel
	Яbias"
_tf_keras_layer
э
!0
"1
12
23
34
45
;6
<7
K8
L9
M10
N11
U12
V13
e14
f15
g16
h17
o18
p19
20
А21
Б22
В23
П24
Р25
Ю26
Я27"
trackable_list_wrapper
ї
!0
"1
12
23
;4
<5
K6
L7
U8
V9
e10
f11
o12
p13
14
А15
П16
Р17
Ю18
Я19"
trackable_list_wrapper
 "
trackable_list_wrapper
ѕ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
√
•trace_0
¶trace_12И
&__inference_3dcnn_layer_call_fn_278253
&__inference_3dcnn_layer_call_fn_278314µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0z¶trace_1
щ
Іtrace_0
®trace_12Њ
A__inference_3dcnn_layer_call_and_return_conditional_losses_278111
A__inference_3dcnn_layer_call_and_return_conditional_losses_278192µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zІtrace_0z®trace_1
ћB…
!__inference__wrapped_model_277658input_1"Ш
С≤Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ђ
©
_variables
™_iterations
Ђ_current_learning_rate
ђ_index_dict
≠
_momentums
Ѓ_velocities
ѓ_update_step_xla"
experimentalOptimizer
-
∞serving_default"
signature_map
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
±non_trainable_variables
≤layers
≥metrics
 іlayer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
г
ґtrace_02ƒ
'__inference_conv3d_layer_call_fn_278449Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zґtrace_0
ю
Јtrace_02я
B__inference_conv3d_layer_call_and_return_conditional_losses_278460Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0
+:)@2conv3d/kernel
:@2conv3d/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Љlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
к
љtrace_02Ћ
.__inference_max_pooling3d_layer_call_fn_278465Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zљtrace_0
Е
Њtrace_02ж
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_278470Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЊtrace_0
<
10
21
32
43"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
њnon_trainable_variables
јlayers
Ѕmetrics
 ¬layer_regularization_losses
√layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
я
ƒtrace_0
≈trace_12§
4__inference_batch_normalization_layer_call_fn_278483
4__inference_batch_normalization_layer_call_fn_278496µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zƒtrace_0z≈trace_1
Х
∆trace_0
«trace_12Џ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278514
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278532µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0z«trace_1
 "
trackable_list_wrapper
':%@2batch_normalization/gamma
&:$@2batch_normalization/beta
/:-@ (2batch_normalization/moving_mean
3:1@ (2#batch_normalization/moving_variance
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
е
Ќtrace_02∆
)__inference_conv3d_1_layer_call_fn_278541Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЌtrace_0
А
ќtrace_02б
D__inference_conv3d_1_layer_call_and_return_conditional_losses_278552Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
-:+@@2conv3d_1/kernel
:@2conv3d_1/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
м
‘trace_02Ќ
0__inference_max_pooling3d_1_layer_call_fn_278557Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0
З
’trace_02и
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_278562Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z’trace_0
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
÷non_trainable_variables
„layers
Ўmetrics
 ўlayer_regularization_losses
Џlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
г
џtrace_0
№trace_12®
6__inference_batch_normalization_1_layer_call_fn_278575
6__inference_batch_normalization_1_layer_call_fn_278588µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zџtrace_0z№trace_1
Щ
Ёtrace_0
ёtrace_12ё
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278606
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278624µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0zёtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
е
дtrace_02∆
)__inference_conv3d_2_layer_call_fn_278633Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
А
еtrace_02б
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278644Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zеtrace_0
.:,@А2conv3d_2/kernel
:А2conv3d_2/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
м
лtrace_02Ќ
0__inference_max_pooling3d_2_layer_call_fn_278649Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zлtrace_0
З
мtrace_02и
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_278654Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zмtrace_0
<
e0
f1
g2
h3"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
г
тtrace_0
уtrace_12®
6__inference_batch_normalization_2_layer_call_fn_278667
6__inference_batch_normalization_2_layer_call_fn_278680µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zтtrace_0zуtrace_1
Щ
фtrace_0
хtrace_12ё
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278698
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278716µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zфtrace_0zхtrace_1
 "
trackable_list_wrapper
*:(А2batch_normalization_2/gamma
):'А2batch_normalization_2/beta
2:0А (2!batch_normalization_2/moving_mean
6:4А (2%batch_normalization_2/moving_variance
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
е
ыtrace_02∆
)__inference_conv3d_3_layer_call_fn_278725Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zыtrace_0
А
ьtrace_02б
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278736Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zьtrace_0
/:-АА2conv3d_3/kernel
:А2conv3d_3/bias
™2І§
Ы≤Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
м
Вtrace_02Ќ
0__inference_max_pooling3d_3_layer_call_fn_278741Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zВtrace_0
З
Гtrace_02и
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_278746Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zГtrace_0
?
0
А1
Б2
В3"
trackable_list_wrapper
/
0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
г
Йtrace_0
Кtrace_12®
6__inference_batch_normalization_3_layer_call_fn_278759
6__inference_batch_normalization_3_layer_call_fn_278772µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЙtrace_0zКtrace_1
Щ
Лtrace_0
Мtrace_12ё
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278790
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278808µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЛtrace_0zМtrace_1
 "
trackable_list_wrapper
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
х
Тtrace_02÷
9__inference_global_average_pooling3d_layer_call_fn_278813Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zТtrace_0
Р
Уtrace_02с
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_278819Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zУtrace_0
0
П0
Р1"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
в
Щtrace_02√
&__inference_dense_layer_call_fn_278828Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЩtrace_0
э
Ъtrace_02ё
A__inference_dense_layer_call_and_return_conditional_losses_278839Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0
 :
АА2dense/kernel
:А2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
ї
†trace_0
°trace_12А
(__inference_dropout_layer_call_fn_278844
(__inference_dropout_layer_call_fn_278849©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z†trace_0z°trace_1
с
Ґtrace_0
£trace_12ґ
C__inference_dropout_layer_call_and_return_conditional_losses_278861
C__inference_dropout_layer_call_and_return_conditional_losses_278866©
Ґ≤Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0z£trace_1
"
_generic_user_object
0
Ю0
Я1"
trackable_list_wrapper
0
Ю0
Я1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
Ш	variables
Щtrainable_variables
Ъregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
д
©trace_02≈
(__inference_dense_1_layer_call_fn_278875Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z©trace_0
€
™trace_02а
C__inference_dense_1_layer_call_and_return_conditional_losses_278886Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
!:	А2dense_1/kernel
:2dense_1/bias
Z
30
41
M2
N3
g4
h5
Б6
В7"
trackable_list_wrapper
Ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
&__inference_3dcnn_layer_call_fn_278253input_1"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
&__inference_3dcnn_layer_call_fn_278314input_1"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
A__inference_3dcnn_layer_call_and_return_conditional_losses_278111input_1"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
A__inference_3dcnn_layer_call_and_return_conditional_losses_278192input_1"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
З
™0
≠1
Ѓ2
ѓ3
∞4
±5
≤6
≥7
і8
µ9
ґ10
Ј11
Є12
є13
Ї14
ї15
Љ16
љ17
Њ18
њ19
ј20
Ѕ21
¬22
√23
ƒ24
≈25
∆26
«27
»28
…29
 30
Ћ31
ћ32
Ќ33
ќ34
ѕ35
–36
—37
“38
”39
‘40"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
 
≠0
ѓ1
±2
≥3
µ4
Ј5
є6
ї7
љ8
њ9
Ѕ10
√11
≈12
«13
…14
Ћ15
Ќ16
ѕ17
—18
”19"
trackable_list_wrapper
 
Ѓ0
∞1
≤2
і3
ґ4
Є5
Ї6
Љ7
Њ8
ј9
¬10
ƒ11
∆12
»13
 14
ћ15
ќ16
–17
“18
‘19"
trackable_list_wrapper
…
’trace_0
÷trace_1
„trace_2
Ўtrace_3
ўtrace_4
Џtrace_5
џtrace_6
№trace_7
Ёtrace_8
ёtrace_9
яtrace_10
аtrace_11
бtrace_12
вtrace_13
гtrace_14
дtrace_15
еtrace_16
жtrace_17
зtrace_18
иtrace_192В
"__inference__update_step_xla_29255
"__inference__update_step_xla_29260
"__inference__update_step_xla_29265
"__inference__update_step_xla_29270
"__inference__update_step_xla_29275
"__inference__update_step_xla_29280
"__inference__update_step_xla_29285
"__inference__update_step_xla_29290
"__inference__update_step_xla_29295
"__inference__update_step_xla_29300
"__inference__update_step_xla_29305
"__inference__update_step_xla_29310
"__inference__update_step_xla_29315
"__inference__update_step_xla_29320
"__inference__update_step_xla_29325
"__inference__update_step_xla_29330
"__inference__update_step_xla_29335
"__inference__update_step_xla_29340
"__inference__update_step_xla_29345
"__inference__update_step_xla_29350ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0z’trace_0z÷trace_1z„trace_2zЎtrace_3zўtrace_4zЏtrace_5zџtrace_6z№trace_7zЁtrace_8zёtrace_9zяtrace_10zаtrace_11zбtrace_12zвtrace_13zгtrace_14zдtrace_15zеtrace_16zжtrace_17zзtrace_18zиtrace_19
–BЌ
$__inference_signature_wrapper_278440input_1"Щ
Т≤О
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ
	jinput_1
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
—Bќ
'__inference_conv3d_layer_call_fn_278449inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
мBй
B__inference_conv3d_layer_call_and_return_conditional_losses_278460inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЎB’
.__inference_max_pooling3d_layer_call_fn_278465inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_278470inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
4__inference_batch_normalization_layer_call_fn_278483inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
4__inference_batch_normalization_layer_call_fn_278496inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278514inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278532inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv3d_1_layer_call_fn_278541inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv3d_1_layer_call_and_return_conditional_losses_278552inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling3d_1_layer_call_fn_278557inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_278562inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
6__inference_batch_normalization_1_layer_call_fn_278575inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
6__inference_batch_normalization_1_layer_call_fn_278588inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278606inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278624inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv3d_2_layer_call_fn_278633inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278644inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling3d_2_layer_call_fn_278649inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_278654inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
6__inference_batch_normalization_2_layer_call_fn_278667inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
6__inference_batch_normalization_2_layer_call_fn_278680inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278698inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278716inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_conv3d_3_layer_call_fn_278725inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278736inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_max_pooling3d_3_layer_call_fn_278741inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_278746inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Б0
В1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
6__inference_batch_normalization_3_layer_call_fn_278759inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
фBс
6__inference_batch_normalization_3_layer_call_fn_278772inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278790inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ПBМ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278808inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
9__inference_global_average_pooling3d_layer_call_fn_278813inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
юBы
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_278819inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
–BЌ
&__inference_dense_layer_call_fn_278828inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
лBи
A__inference_dense_layer_call_and_return_conditional_losses_278839inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBџ
(__inference_dropout_layer_call_fn_278844inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ёBџ
(__inference_dropout_layer_call_fn_278849inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
C__inference_dropout_layer_call_and_return_conditional_losses_278861inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
C__inference_dropout_layer_call_and_return_conditional_losses_278866inputs"§
Э≤Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
“Bѕ
(__inference_dense_1_layer_call_fn_278875inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
C__inference_dense_1_layer_call_and_return_conditional_losses_278886inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
й	variables
к	keras_api

лtotal

мcount"
_tf_keras_metric
c
н	variables
о	keras_api

пtotal

рcount
с
_fn_kwargs"
_tf_keras_metric
0:.@2Adam/m/conv3d/kernel
0:.@2Adam/v/conv3d/kernel
:@2Adam/m/conv3d/bias
:@2Adam/v/conv3d/bias
,:*@2 Adam/m/batch_normalization/gamma
,:*@2 Adam/v/batch_normalization/gamma
+:)@2Adam/m/batch_normalization/beta
+:)@2Adam/v/batch_normalization/beta
2:0@@2Adam/m/conv3d_1/kernel
2:0@@2Adam/v/conv3d_1/kernel
 :@2Adam/m/conv3d_1/bias
 :@2Adam/v/conv3d_1/bias
.:,@2"Adam/m/batch_normalization_1/gamma
.:,@2"Adam/v/batch_normalization_1/gamma
-:+@2!Adam/m/batch_normalization_1/beta
-:+@2!Adam/v/batch_normalization_1/beta
3:1@А2Adam/m/conv3d_2/kernel
3:1@А2Adam/v/conv3d_2/kernel
!:А2Adam/m/conv3d_2/bias
!:А2Adam/v/conv3d_2/bias
/:-А2"Adam/m/batch_normalization_2/gamma
/:-А2"Adam/v/batch_normalization_2/gamma
.:,А2!Adam/m/batch_normalization_2/beta
.:,А2!Adam/v/batch_normalization_2/beta
4:2АА2Adam/m/conv3d_3/kernel
4:2АА2Adam/v/conv3d_3/kernel
!:А2Adam/m/conv3d_3/bias
!:А2Adam/v/conv3d_3/bias
/:-А2"Adam/m/batch_normalization_3/gamma
/:-А2"Adam/v/batch_normalization_3/gamma
.:,А2!Adam/m/batch_normalization_3/beta
.:,А2!Adam/v/batch_normalization_3/beta
%:#
АА2Adam/m/dense/kernel
%:#
АА2Adam/v/dense/kernel
:А2Adam/m/dense/bias
:А2Adam/v/dense/bias
&:$	А2Adam/m/dense_1/kernel
&:$	А2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
нBк
"__inference__update_step_xla_29255gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29260gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29265gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29270gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29275gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29280gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29285gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29290gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29295gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29300gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29305gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29310gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29315gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29320gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29325gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29330gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29335gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29340gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29345gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
"__inference__update_step_xla_29350gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
л0
м1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
:  (2total
:  (2count
0
п0
р1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperб
A__inference_3dcnn_layer_call_and_return_conditional_losses_278111Ы#!"1234;<KLMNUVefghopАБВПРЮЯFҐC
<Ґ9
/К,
input_1€€€€€€€€€АА@
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ б
A__inference_3dcnn_layer_call_and_return_conditional_losses_278192Ы#!"1234;<KLMNUVefghopАБВПРЮЯFҐC
<Ґ9
/К,
input_1€€€€€€€€€АА@
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ї
&__inference_3dcnn_layer_call_fn_278253Р#!"1234;<KLMNUVefghopАБВПРЮЯFҐC
<Ґ9
/К,
input_1€€€€€€€€€АА@
p

 
™ "!К
unknown€€€€€€€€€ї
&__inference_3dcnn_layer_call_fn_278314Р#!"1234;<KLMNUVefghopАБВПРЮЯFҐC
<Ґ9
/К,
input_1€€€€€€€€€АА@
p 

 
™ "!К
unknown€€€€€€€€€Ѓ
"__inference__update_step_xla_29255ЗАҐ}
vҐs
%К"
gradient@
@Т=	)Ґ&
ъ@
А
p
` VariableSpec 
`аЦўъҐЏ?
™ "
 М
"__inference__update_step_xla_29260f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`аъҐшҐЏ?
™ "
 М
"__inference__update_step_xla_29265f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`аФтііЏ?
™ "
 М
"__inference__update_step_xla_29270f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`јУтііЏ?
™ "
 Ѓ
"__inference__update_step_xla_29275ЗАҐ}
vҐs
%К"
gradient@@
@Т=	)Ґ&
ъ@@
А
p
` VariableSpec 
`†…тііЏ?
™ "
 М
"__inference__update_step_xla_29280f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`А√тііЏ?
™ "
 М
"__inference__update_step_xla_29285f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†£уііЏ?
™ "
 М
"__inference__update_step_xla_29290f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`АҐуііЏ?
™ "
 ∞
"__inference__update_step_xla_29295ЙВҐ
xҐu
&К#
gradient@А
AТ>	*Ґ'
ъ@А
А
p
` VariableSpec 
`†®уііЏ?
™ "
 О
"__inference__update_step_xla_29300hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`а™уііЏ?
™ "
 О
"__inference__update_step_xla_29305hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`јДршҐЏ?
™ "
 О
"__inference__update_step_xla_29310hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`†–уііЏ?
™ "
 ≥
"__inference__update_step_xla_29315МЕҐБ
zҐw
'К$
gradientАА
BТ?	+Ґ(
ъАА
А
p
` VariableSpec 
`АМршҐЏ?
™ "
 О
"__inference__update_step_xla_29320hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`јОршҐЏ?
™ "
 О
"__inference__update_step_xla_29325hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`АжршҐЏ?
™ "
 О
"__inference__update_step_xla_29330hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`јЄуііЏ?
™ "
 Ш
"__inference__update_step_xla_29335rlҐi
bҐ_
К
gradient
АА
6Т3	Ґ
ъ
АА
А
p
` VariableSpec 
`јЖсшҐЏ?
™ "
 О
"__inference__update_step_xla_29340hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`аіэцҐЏ?
™ "
 Ц
"__inference__update_step_xla_29345pjҐg
`Ґ]
К
gradient	А
5Т2	Ґ
ъ	А
А
p
` VariableSpec 
`АїсшҐЏ?
™ "
 М
"__inference__update_step_xla_29350f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`аѓсшҐЏ?
™ "
 Њ
!__inference__wrapped_model_277658Ш#!"1234;<KLMNUVefghopАБВПРЮЯ>Ґ;
4Ґ1
/К,
input_1€€€€€€€€€АА@
™ "1™.
,
dense_1!К
dense_1€€€€€€€€€С
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278606їKLMN^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "SҐP
IКF
tensor_08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ С
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_278624їKLMN^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "SҐP
IКF
tensor_08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ л
6__inference_batch_normalization_1_layer_call_fn_278575∞KLMN^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "HКE
unknown8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@л
6__inference_batch_normalization_1_layer_call_fn_278588∞KLMN^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "HКE
unknown8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@У
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278698љefgh_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p

 
™ "TҐQ
JКG
tensor_09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ У
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_278716љefgh_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 

 
™ "TҐQ
JКG
tensor_09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ н
6__inference_batch_normalization_2_layer_call_fn_278667≤efgh_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p

 
™ "IКF
unknown9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ан
6__inference_batch_normalization_2_layer_call_fn_278680≤efgh_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 

 
™ "IКF
unknown9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АЦ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278790јАБВ_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p

 
™ "TҐQ
JКG
tensor_09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ц
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_278808јАБВ_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 

 
™ "TҐQ
JКG
tensor_09€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ р
6__inference_batch_normalization_3_layer_call_fn_278759µАБВ_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p

 
™ "IКF
unknown9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Ар
6__inference_batch_normalization_3_layer_call_fn_278772µАБВ_Ґ\
UҐR
HКE
inputs9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 

 
™ "IКF
unknown9€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278514ї1234^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "SҐP
IКF
tensor_08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ П
O__inference_batch_normalization_layer_call_and_return_conditional_losses_278532ї1234^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "SҐP
IКF
tensor_08€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ й
4__inference_batch_normalization_layer_call_fn_278483∞1234^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p

 
™ "HКE
unknown8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@й
4__inference_batch_normalization_layer_call_fn_278496∞1234^Ґ[
TҐQ
GКD
inputs8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 

 
™ "HКE
unknown8€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€@√
D__inference_conv3d_1_layer_call_and_return_conditional_losses_278552{;<;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€??@
™ "8Ґ5
.К+
tensor_0€€€€€€€€€==@
Ъ Э
)__inference_conv3d_1_layer_call_fn_278541p;<;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€??@
™ "-К*
unknown€€€€€€€€€==@ƒ
D__inference_conv3d_2_layer_call_and_return_conditional_losses_278644|UV;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ "9Ґ6
/К,
tensor_0€€€€€€€€€А
Ъ Ю
)__inference_conv3d_2_layer_call_fn_278633qUV;Ґ8
1Ґ.
,К)
inputs€€€€€€€€€@
™ ".К+
unknown€€€€€€€€€А≈
D__inference_conv3d_3_layer_call_and_return_conditional_losses_278736}op<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ "9Ґ6
/К,
tensor_0€€€€€€€€€А
Ъ Я
)__inference_conv3d_3_layer_call_fn_278725rop<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€А
™ ".К+
unknown€€€€€€€€€А√
B__inference_conv3d_layer_call_and_return_conditional_losses_278460}!"=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€АА@
™ "8Ґ5
.К+
tensor_0€€€€€€€€€~~>@
Ъ Э
'__inference_conv3d_layer_call_fn_278449r!"=Ґ:
3Ґ0
.К+
inputs€€€€€€€€€АА@
™ "-К*
unknown€€€€€€€€€~~>@≠
C__inference_dense_1_layer_call_and_return_conditional_losses_278886fЮЯ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ З
(__inference_dense_1_layer_call_fn_278875[ЮЯ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "!К
unknown€€€€€€€€€ђ
A__inference_dense_layer_call_and_return_conditional_losses_278839gПР0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
&__inference_dense_layer_call_fn_278828\ПР0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ""К
unknown€€€€€€€€€Ађ
C__inference_dropout_layer_call_and_return_conditional_losses_278861e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ ђ
C__inference_dropout_layer_call_and_return_conditional_losses_278866e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
(__inference_dropout_layer_call_fn_278844Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ ""К
unknown€€€€€€€€€АЖ
(__inference_dropout_layer_call_fn_278849Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ ""К
unknown€€€€€€€€€Ас
T__inference_global_average_pooling3d_layer_call_and_return_conditional_losses_278819Ш_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "5Ґ2
+К(
tensor_0€€€€€€€€€€€€€€€€€€
Ъ Ћ
9__inference_global_average_pooling3d_layer_call_fn_278813Н_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "*К'
unknown€€€€€€€€€€€€€€€€€€П
K__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_278562њ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "\ҐY
RКO
tensor_0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ й
0__inference_max_pooling3d_1_layer_call_fn_278557і_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "QКN
unknownA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
K__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_278654њ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "\ҐY
RКO
tensor_0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ й
0__inference_max_pooling3d_2_layer_call_fn_278649і_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "QКN
unknownA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
K__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_278746њ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "\ҐY
RКO
tensor_0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ й
0__inference_max_pooling3d_3_layer_call_fn_278741і_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "QКN
unknownA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
I__inference_max_pooling3d_layer_call_and_return_conditional_losses_278470њ_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "\ҐY
RКO
tensor_0A€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ з
.__inference_max_pooling3d_layer_call_fn_278465і_Ґ\
UҐR
PКM
inputsA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "QКN
unknownA€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ћ
$__inference_signature_wrapper_278440£#!"1234;<KLMNUVefghopАБВПРЮЯIҐF
Ґ 
?™<
:
input_1/К,
input_1€€€€€€€€€АА@"1™.
,
dense_1!К
dense_1€€€€€€€€€