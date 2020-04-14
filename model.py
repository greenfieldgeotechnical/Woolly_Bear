def woolyTrain():
    '''Trains two models:
        1.  SVC with PCA
        2.  k nearest neighbors/
        
    Then does a prediction afterwards''' 
    
    import pickle
    import sklearn.model_selection as model_selection
    import sklearn.decomposition as decomposition
    import sklearn.preprocessing as preprocessing
    import sklearn.neighbors as neighbors
    import sklearn.svm as svm
    import PIL
    import os
    import numpy
    import scipy.stats as stats
    
    
    d = 'Photos\\Processed'
    with open('y.dat','rb') as f:
        dc = pickle.load(f)
    
    y = []
    c = []
    X = []
    
    normi = stats.norm.ppf(0.5+(1./6.))
    
    for k in dc:
#        yi = (k['air'] - k['air_mean'])/k['air_std']
        yi = (k['precip'] - k['mean'])/k['std']
        
        if yi <= -normi:
            ci = 'below'
        elif yi <= normi:
            ci = 'normal'
        else:
            ci = 'above'
        
        im = PIL.Image.open(os.path.join(d,k['photo']))
        
        xi = numpy.array(im)
        h,w,nrgb = numpy.shape(xi)
            
        X.append(xi.flatten())
        y.append(yi)
        c.append(ci)
        
    #split!
    Xt,Xs,ct,cs = model_selection.train_test_split(X,c,test_size=0.10)

    #PCA.  100 components
    n = min(100, len(Xt))
    pca = decomposition.KernelPCA(n_components=n, kernel='rbf')
    pca.fit(Xt)
    
    Xtp = pca.transform(Xt)
    Xsp = pca.transform(Xs)
    
    #sklearn!  SVC
    trans = preprocessing.QuantileTransformer(output_distribution='normal')
    Xtpn = trans.fit_transform(Xtp)
    Xspn = trans.transform(Xsp)
    
    svc = svm.SVC(kernel='rbf',gamma='auto', C=1., probability=True)
    svc.fit(Xtpn,ct)
    sc = [svc.score(Xtpn, ct), svc.score(Xspn,cs)] #perfect fit, duh
    print('SVC with PCA:  {:.0%} training, {:.0%} test'.format(*sc))
    
    #huge file for some reason, return the best model for processing
    #dump trans, svc, ct, Xtpn, and then fit in function.
    df = {'svc':svc, 'trans': trans, 'pca':pca}
    with open('precip model.dat','wb') as f:
        pickle.dump(df,f)
    
    #sklearn! Nearest neighbor
    neigh = neighbors.KNeighborsClassifier()
    neigh.fit(Xt,ct)
    sn = [neigh.score(Xt,ct), neigh.score(Xs,cs)] #perfect fit, duh
    print('Nearest neighbor:  {:.0%} training, {:.0%} test'.format(*sn))
    
    print('Returning models')
    #huge file for some reason, return the best model for processing
    return svc, neigh, pca, trans

def woolyid():
    
    #train a tensorflow model
    import PIL
    import os
    import numpy
    import scipy.stats as stats
    import sklearn.model_selection as model_selection
    import sklearn.preprocessing as preprocessing
    from tensorflow import keras
    
    d = 'Photos\\Processed'
    df = 'Photos\\fakes\\Processed'
    
    c = []
    X = []
    
    #start with real photos
    for k in os.listdir(d):
        try:
            im = PIL.Image.open(os.path.join(d,k))
        except:
            continue
        
        xi = numpy.array(im)
        h,w,nrgb = numpy.shape(xi)
            
        X.append(xi)
        c.append('Yes')
        
    for k in os.listdir(df):
        try:
            im = PIL.Image.open(os.path.join(df,k))
        except:
            continue
        
        xi = numpy.array(im)
        h,w,nrgb = numpy.shape(xi)
            
        X.append(xi)
        c.append('No')
        
    #split!
    Xt,Xs,ct,cs = model_selection.train_test_split(X,c,test_size=0.10)  #no it usually zero?
    le = preprocessing.LabelEncoder()
    le.fit(numpy.unique(ct))
    it = le.transform(ct)
    iss = le.transform(cs)
    
    nc = len(numpy.unique(c))
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(h,w,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Dropout(0.5))  #doesn't save space!
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))  #only 2 classifications
    
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    Xt = numpy.array(Xt)
    Xs = numpy.array(Xs)
    model.fit(Xt,it,epochs=10)
    
    model.evaluate(Xs, iss)
    
    model.predict(Xs)
    
    #save model
    model_json = model.to_json()
    with open(os.path.join('model',"woollyID.json"), "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights(os.path.join('model',"woollyID.h5"))
    
#    #convert it to tensorflow lite. Maybe necessary if server cant handle keras
#    import tensorflow
#    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
#    tflite_model = converter.convert()
#    
#    with open(os.path.join('model','id.tflite'),'wb') as f:
#        f.write(tflite_model)
        

def clim():
    #data from https://www.esrl.noaa.gov/psd/data/gridded/data.cmap.html
    #air temp from https://www.esrl.noaa.gov/psd/data/gridded/data.ghcncams.html
    import xarray
    import numpy
    import openpyxl
    import datetime
    import pandas
    import pickle
    
    dfile = 'Climate Data\\precip.mon.mean.nc'
    data = xarray.open_dataset(dfile)
    
    dfile = 'Climate Data\\air.mon.mean.nc'
    datat = xarray.open_dataset(dfile)

    #go get the metadata
    wb = openpyxl.load_workbook('Metadata.xlsx')
    s = wb['Metadata']
    d = []
    #make an list of lists of the excel sheet
    for r in s.rows:
        x = []
        for c in r:            
            x.append(c.value)
        d.append(x)
        
    #load the climate data and make a dict.  Important parameters!
    w = ['11-01-','04-01-']
    
    yrs = numpy.unique([k.year for k in pandas.DatetimeIndex(data.time.values)])
    dc = []
    di = {}
    for k in d:
        #check it it's got everything
        if len(k) < 4:
            continue
        elif type(k[1]) != datetime.datetime:
            continue
        elif k[1].year == 2019:   #update for this year!
            continue
        elif k[2] == None:
            continue
            
        c = data.sel(lat=k[2], lon=360.+k[3],method='nearest').to_dataframe()
        ct = datat.sel(lat=k[2], lon=360.+k[3],method='nearest').to_dataframe()
        
        #get precip data over that period
        p = {}
        pt = {}
        for j in yrs[:-1]:
            cm = c.loc[pandas.date_range(start=w[0]+str(j), end=w[1]+str(j+1))]
            p.update({j:numpy.sum(cm.precip)})
            cmt = ct.loc[pandas.date_range(start=w[0]+str(j), end=w[1]+str(j+1))]
            pt.update({j:numpy.mean(cmt.air - 273.15)})
        
        
        di = {'photo':k[0], 'time':k[1], 'lat':k[2], 'lon':k[3], 'mean':numpy.mean(list(p.values())), 'std':numpy.std(list(p.values())), 'precip':p[k[1].year],
            'air_mean': numpy.mean(list(pt.values())), 'air_std': numpy.std(list(pt.values())), 'air': pt[k[1].year]}
        dc.append(di.copy())
        
    with open('y.dat','wb') as f:
        pickle.dump(dc,f)
    
def photoReshape():
    import PIL
    import os
    
    #format
    w = 640
    h = 480
    
    d = 'D:\Dropbox\Machine Learning\Wooly bear\Photos'
    fn = os.listdir(d)
    fj = [k for k in fn if os.path.splitext(k)[1].lower() == '.jpg' or os.path.splitext(k)[1].lower() == '.jpeg']
    for k in fj:
        im = PIL.Image.open(os.path.join(d,k))
        wi,hi = im.size
        ri = float(wi)/float(hi)
        
        if ri < 1.:
            #just flip it
            im = im.transpose(PIL.Image.ROTATE_90)
            wi,hi = im.size
            ri = float(wi)/float(hi)
            
        #processing.  Crop
        if ri > 4./3.:
            #too wide
            wt = int(float(hi)*4./3.)
            off = int((wi-wt)/2.)
            im = im.crop([off,0,wi-off,hi])
        elif ri < 4./3.:
            #too tall
            ht = int(float(wi)*3./4.)
            off = int((hi-ht)/2.)
            im = im.crop([0,off,wi,hi-off])
            
        im = im.resize([w,h])
            
        #save the image
        im.save(os.path.join(d,'Processed',k))
        
def photoReshapei(fn='..\\static\\img\\1463292773_87d3868b66_z.jpg'):
    import PIL.Image as Image
    import os
    
    #format
    w = 640
    h = 480
    
    d,fi = os.path.split(fn)

    im = Image.open(fn)
    
    wi,hi = im.size
    ri = float(wi)/float(hi)
    
    flip = False
    if ri < 1.:
        #just flip it
        im = im.transpose(Image.ROTATE_90)
        wi,hi = im.size
        ri = float(wi)/float(hi)
        flip = True
        
    #processing.  Crop
    if ri > 4./3.:
        #too wide
        wt = int(float(hi)*4./3.)
        off = int((wi-wt)/2.)
        im = im.crop([off,0,wi-off,hi])
    elif ri < 4./3.:
        #too tall
        ht = int(float(wi)*3./4.)
        off = int((hi-ht)/2.)
        im = im.crop([0,off,wi,hi-off])
        
    im = im.resize([w,h])
        
    #save the image
    im.save(os.path.join(d,'Processed',fi))
    
    return flip
    
    
            
    