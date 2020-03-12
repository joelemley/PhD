from scipy import io

data=""

person="00"

samples=[]

with open ("MPIIGaze/Evaluation Subset/p"+person+".txt", "r") as myfile:
    data = myfile.readlines()

for sampleid in xrange(0,len(data)):

    day= data[sampleid][0:5]
    whicheye=data[sampleid][15:].strip()
    dataindex=int(data[sampleid][6:10])-1# the -1 is to correct for matlab indexing. 0001 becomes 0000

    matlocation="MPIIGaze/Data/Normalized/p"+person+"/"+day+".mat"

    matfile=io.loadmat(matlocation)

    image=matfile["data"][whicheye][0][0][0][0]["image"][dataindex]
    gaze=matfile["data"][whicheye][0][0][0][0]["pose"][dataindex]
    pose=matfile["data"][whicheye][0][0][0][0]["gaze"][dataindex]
    d={'eye': whicheye, 'pose':pose, 'gaze': gaze, 'image': image}
    samples.append(d)
    print sampleid
io.savemat("p00",)
print "hello"