def upload(iswb, fn = 'static\\img\\Processed\\8987134764_5a7ea96a14_z.jpg', ipdata={}, userdata={}):
    import boto3
    import os
    import json

    sess = boto3.session.Session()
    s3 = sess.client(
            service_name='s3',
            region_name='us-east-2'
    )

    fni = os.path.split(fn)
    objn = os.path.split(fn)[-1]
    _ = s3.upload_file(fn, 'woolly-bear', 'photos/'+objn)

    if iswb:
        _ = s3.upload_file(os.path.join(fni[0],'output',fni[1]), 'woolly-bear', 'photos/output/'+objn)

        with open('latest.txt', 'w') as f:
            f.write(objn)

        _ = s3.upload_file('latest.txt', 'woolly-bear', 'latest.txt')

    ipdata.update(userdata)
    ipdata.update({'iswb':bool(iswb)})

    ipj = json.dumps(ipdata)
    with open(os.path.join('static','ipdata',os.path.splitext(objn)[0] + '.dat'), 'w') as f:
        f.write(ipj)

    _ = s3.upload_file(os.path.join('static','ipdata',os.path.splitext(objn)[0] + '.dat'), 'woolly-bear', 'ipdata/'+os.path.splitext(objn)[0] + '.dat')


def screen(fn = 'static\\img\\Processed\\4043336973_2e6f6b4ce1_z.jpg', flip=False):
    import PIL
    import numpy
    from PIL import ImageFont
    from PIL import ImageDraw
    import os

    im = PIL.Image.open(fn)

    xi = numpy.array(im, dtype=float)
    h,w,nrgb = numpy.shape(xi)

    #screening model keras
    from tensorflow import keras
    with open(os.path.join('model','woollyID.json'), 'r') as json_file:
        model_json = json_file.read()

    model = keras.models.model_from_json(model_json)
    model.load_weights(os.path.join('model','woollyID.h5'))

    prob = model.predict(xi.reshape(1,h,w,3))  #['No','Yes']
    iswb = model.predict_classes(xi.reshape(1,h,w,3))

    if not iswb:
        out = 'This is not a woolly bear!'
        if flip:
            im = im.transpose(PIL.Image.ROTATE_270)
            h0 = h
            h = w
            w = h0
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("static/fonts/arialbi.ttf", 32)
        wt,ht = draw.textsize(out)
        draw.text(((w-wt)/4,int(h*0.65)),out,(255,255,255),font=font)

        fni = os.path.split(fn)
        im.save(os.path.join(fni[0],'output',fni[1]))

    return iswb

def predict(fn = 'static\\img\\Processed\\1463292773_87d3868b66_z.jpg', flip=False, ipdata={}):
    import PIL
    import numpy
    import pickle
    from PIL import ImageFont
    from PIL import ImageDraw
    import os
    import boto3
    import json


    md = numpy.random.choice(['air','precip'])
    mdf = md + ' model.dat'
    mdl = {'air': ['warmer','colder'], 'precip': ['wetter', 'drier']}

    rk = {'above':'{:} than average'.format(mdl[md][0]),
          'below': '{:} than average'.format(mdl[md][1]),
          'normal': 'average'}

    #saved as air
    with open(os.path.join('model',mdf),'rb') as f:
        df = pickle.load(f)

    svc = df['svc']
    trans = df['trans']
    pca = df['pca']

    im = PIL.Image.open(fn)

    x = numpy.array(im)
    h,w,nrgb = numpy.shape(x)
    xi = x.flatten()

    xt = pca.transform(xi.reshape(1,-1))
    xtn = trans.transform(xt)
    cp = svc.predict_proba(xtn)[0]

    res = dict(zip(svc.classes_,cp))
    resk = dict([(rk[i],v) for i,v in res.items()])
    v = max(resk, key=resk.get)
    p = int(resk[v]*100.)
    try:
        out = '{:}% probability of a \n  {:} winter in \n  {:}, {:}'.format(p,v, ipdata['city'], ipdata['region_name'])
    except:
        out = '{:}% probability of a \n   {:} winter'.format(p,v)

    if flip:
        im = im.transpose(PIL.Image.ROTATE_270)
        h0 = h
        h = w
        w = h0
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("static/fonts/arialbi.ttf", 32)
    wt,ht = draw.textsize(out)
    draw.text(((w-wt)/4,int(h*0.55)),out,(255,255,255),font=font)

    fni = os.path.split(fn)
    im.save(os.path.join(fni[0],'output',fni[1]))

    return res


def rgbImg():
    import PIL
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt

    fn = 'Photos\\Processed\\30199289921_54b18bb299_z.jpg'
    im = PIL.Image.open(fn)

    xi = numpy.array(im)
    h,w,nrgb = numpy.shape(xi)

    fig = plt.figure(1, figsize=(4,3))
    plt.clf()
    font = {'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 16}
    matplotlib.rc('font', **font)

    plt.imshow(xi[:,:,0], cmap='Reds')
    plt.axis('off')
    plt.savefig('Posts\\Red.jpg', dpi=400)

    # plt.imshow(xi[:,:,1], cmap='Greens')

    plt.imshow(xi[:,:,2], cmap='Blues')
    plt.axis('off')
    plt.savefig('Posts\\Blue.jpg', dpi=400)

    plt.imshow(xi[:,:,0] - xi[:,:,2], cmap='cool')

def eigen():

    import PIL
    from PIL import ImageEnhance
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    import sklearn.decomposition as decomposition
    import os

    d = '../Processed'
    #fn = [k for k in os.listdir('Photos\\Processed') if os.path.splitext(k)[-1].lower()=='.jpg']

    fn = ['30199289921_54b18bb299_z.jpg',
          '10200080326_60cac3dc39_z.jpg']

    n = len(fn)
    X = []
    for k in range(n):
        im = PIL.Image.open(os.path.join(d,fn[k]))
        xi = numpy.array(im)
        h,w,nrgb = numpy.shape(xi)
        X.append(xi[:,:,:].flatten())

    X = numpy.array(X)
    pca = decomposition.PCA(n_components=n, svd_solver='randomized')
    p = pca.fit(numpy.array(X))
    eb = p.components_.reshape(n,h,w,nrgb)

    # plt.figure(1)
    # plt.axis('off')
    # plt.imshow(X[0,:].reshape(h,w,nrgb))

    plt.figure(2)
    plt.clf()
    # y = eb[0,:,:,:]  #47
    y = eb[0,:,:,2]  #47
    yt = ((y - min(y.flatten()))/(max(y.flatten()) - min(y.flatten())))*255
    yi = yt
    # yi = numpy.min(yt, axis=2)
    yt[yi>80] = 0
    # yt[yi<100] = 255
    im = PIL.Image.fromarray(numpy.uint8(yt))
    imc = ImageEnhance.Brightness(im).enhance(1.0)
    # imc = ImageEnhance.Color(imc).enhance(1.5)
    # imc = ImageEnhance.Contrast(imc).enhance(1.0)
    # imc = ImageEnhance.Sharpness(imc).enhance(0.5)
    plt.imshow(imc, cmap='Greys')
    plt.axis('off')
    plt.savefig('eigen.jpg')

    return y

    #8987134764_5a7ea96a14_z
    #10256117936_ab539d1d79_z



def map():
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime
    import pandas
    import numpy
    import xarray
    import matplotlib.colors as colors


    fig1 = plt.figure(1, figsize=(8.,5.))
    ax = plt.gca()
    plt.clf()
    font = {'family' : 'Arial',
                'weight' : 'normal',
                'size'   : 24}
    matplotlib.rc('font', **font)

    ax = plt.subplot(111, projection=ccrs.Mercator(central_longitude=-100., min_latitude=32., max_latitude=57.))

    ax.set_extent([-130., -65, 30, 55.])
    states_shp = shpreader.natural_earth(resolution='50m',
                                         category='cultural', name='admin_1_states_provinces_shp')

    for state in shpreader.Reader(states_shp).geometries():
        # pick a default color for the land with a black outline,
        # this will change if the storm intersects with our track
        facecolor = [0.95, 0.95, 0.95]
        edgecolor = 'black'

        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor, zorder=0)

    #ocean for masking
    ocean_shp = shpreader.natural_earth(resolution='50m',
                                         category='physical', name='ocean')

    for ocean in shpreader.Reader(ocean_shp).geometries():
        # pick a default color for the land with a black outline,
        # this will change if the storm intersects with our track
        facecolor = [0.0, 0.0, 0.]
        edgecolor = 'black'

        ax.add_geometries([ocean], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor, alpha = 1.0, zorder=2)
        ax.add_geometries([ocean], ccrs.PlateCarree(),
                          facecolor='w', edgecolor='k', alpha = 0.6, zorder=3)

    #great lakes because it looks cool
    lakes_shp = shpreader.natural_earth(resolution='50m',
                category='physical', name='lakes')

    for lakes in shpreader.Reader(lakes_shp).geometries():
        # pick a default color for the land with a black outline,
        # this will change if the storm intersects with our track
        facecolor = [0.0, 0.0, 0.]
        edgecolor = 'black'

        ax.add_geometries([lakes], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor, alpha = 1.0, zorder=2)
        ax.add_geometries([lakes], ccrs.PlateCarree(),
                          facecolor='w', edgecolor='k', alpha = 0.6, zorder=3)


    dfile = 'Climate Data\\air.mon.mean.nc'
    data = xarray.open_dataset(dfile)

    #average for three months
    w = ['11-01-','01-01-']
    xt = data.coords['lon'].values-180.
    yt = data.coords['lat'].values
    x = xt[(xt >= -131.)&(xt <= -64.)]
    y = yt[(yt >= 30.)&(yt <= 58.)]
    nx = len(x)
    ny = len(y)

    #load the climate data and make a dict.  Important parameters!
    w = ['11-01-','01-01-']

    yrs = numpy.unique([k.year for k in pandas.DatetimeIndex(data.time.values)])

    z = numpy.zeros([nx,ny])
    zc = numpy.zeros([nx,ny])
    zsig = numpy.zeros([nx,ny])

    #split data into n years
    s = {}
    for k in yrs[:-1]:
        s.update({k:(data.sel(time = pandas.date_range(start=w[0]+str(k), end=w[1]+str(k+1), freq='MS'), method='nearest'))})


    for i in range(nx):
        print('{:} of {:}'.format(i,nx))
        for j in range(ny):

            #s = data.sel(lat=y[j], lon=x[i]+180.).to_dataframe()

            p = []
            for k in yrs[:-1]:
                cm = s[k].sel(lat=y[j], lon=x[i]+180.).air.values - 273.15
                #cm = s[k].loc[pandas.date_range(start=w[0]+str(k), end=w[1]+str(k+1))]
                p.append(numpy.mean(cm))

            z[i,j] = numpy.mean(p[:-1])
            zsig[i,j] = numpy.std(p[:-1])
            zc[i,j] = p[-1]


    zd = zc - z
    zd[numpy.isnan(zd)] = 0.
    levels = numpy.linspace(-4.,4.,81)
    h = plt.contourf(x,y,zd.T,cmap = 'seismic', alpha = 0.8, zorder=2, levels=levels, transform=ccrs.PlateCarree(), norm=colors.Normalize(vmin=-4., vmax=4.))
    for c in h.collections:
        c.set_edgecolor('face')
        c.set_linewidth(0.)
        c.set_alpha(0.8)

    #inches of precip
    gsa = fig1.add_axes([0.84,0.17, 0.02, 0.2])
    cbar = fig1.colorbar(h, cax = gsa, ticks = numpy.linspace(-4.,4.,5), orientation='vertical', extend='max')
    gsa.tick_params(labelsize=12)
    gsa.set_ylabel('Difference \n'+r'from avg. ($\rm{^o}$C)', fontsize=12)
    gsa.yaxis.set_label_position('left')


    plt.savefig('Posts\\Deg Jan 2019.jpg', dpi=600)

    print(numpy.mean(zd))

def hist():
    import matplotlib.pyplot as plt
    import matplotlib

    fig1 = plt.figure(1, figsize=(8.,5.))
    ax = plt.gca()
    plt.clf()
    font = {'family' : 'Arial',
                'weight' : 'normal',
                'size'   : 24}
    matplotlib.rc('font', **font)
    cm = matplotlib.cm.viridis_r
    plt.clf()


    c = ['drier', 'normal', 'wetter']
    x = [0,1,2]
    p = [0.36237958, 0.53691153, 0.1007089 ]
    colors = cm([0.1, 0.5, 0.9])

    plt.bar(x,p,0.6, color=colors, alpha=0.8,)
    plt.ylim([0,1])
    plt.xticks(x,c,rotation=40)
    plt.ylabel('Probability')
    plt.tight_layout()

    plt.savefig('Posts\\hist.jpg',dpi=300)
