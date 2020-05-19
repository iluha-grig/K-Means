import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import random
import math

def dist(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

dataset = np.random.uniform(0, 25, 200)
dataset = np.append(dataset, np.random.uniform(10, 70, 400))
dataset = np.append(dataset, np.random.uniform(30, 50, 100))
np.random.shuffle(dataset)
dataset = dataset.reshape(-1, 2)
eps = 0.0001

fig1 = plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
xpoints, ypoints = np.hsplit(dataset, 2)
plt.scatter(xpoints, ypoints, c = 'black', s = 1)
plt.savefig('input.png', dpi = 800, format = 'png')

print('Input the quantity of clasters from 2 to 4 or print "auto" to choose automatically...')
s = input()

if s == '2':
    size = dataset.shape
    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    fclaster = 0
    sclaster = 0
    fcenter = 0
    scenter = 0
    flag1 = 1
    flag2 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            if fdist < sdist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif fdist > sdist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            else:
                if random.randint(0, 1):
                    if flag2:
                        sclaster = np.array(dataset[i])
                        flag2 = 0
                    else:
                        sclaster = np.vstack((sclaster, dataset[i]))
                else:
                    if flag1:
                        fclaster = np.array(dataset[i])
                        flag1 = 0
                    else:
                        fclaster = np.vstack((fclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps):
            break
        first = fcenter
        second = scenter
        flag1 = 1
        flag2 = 1
        fclaster = 0
        sclaster = 0
        fcenter = 0
        scenter = 0

    fig2 = plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    xpoints, ypoints = np.hsplit(fclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'red', s = 1)
    xpoints, ypoints = np.hsplit(sclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
    plt.savefig('clastered_2.png', dpi = 800, format = 'png')

if s == '3':
    size = dataset.shape
    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i])) / 2) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i])) / 2
            third = dataset[i]
    fclaster = 0
    sclaster = 0
    tclaster = 0
    fcenter = 0
    scenter = 0
    tcenter = 0
    flag1 = 1
    flag2 = 1
    flag3 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            tdist = dist(third, dataset[i])
            mindist = min(fdist, sdist, tdist)
            if fdist == mindist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif sdist == mindist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            else:
                if flag3:
                    tclaster = np.array(dataset[i])
                    flag3 = 0
                else:
                    tclaster = np.vstack((tclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        for i in np.arange(tclaster.shape[0]):
            tcenter = tcenter + tclaster[i]
        tcenter = tcenter / tclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps) and (dist(third, tcenter) < eps):
            break
        first = fcenter
        second = scenter
        third = tcenter
        flag1 = 1
        flag2 = 1
        flag3 = 1
        fclaster = 0
        sclaster = 0
        tclaster = 0
        fcenter = 0
        scenter = 0
        tcenter = 0

    fig2 = plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    xpoints, ypoints = np.hsplit(fclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'red', s = 1)
    xpoints, ypoints = np.hsplit(sclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
    xpoints, ypoints = np.hsplit(tclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'green', s = 1)
    plt.savefig('clastered_3.png', dpi = 800, format = 'png')

if s == '4':
    size = dataset.shape
    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i])) / 2) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i])) / 2
            third = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i]) + dist(third, dataset[i])) / 3) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i]) + dist(third, dataset[i])) / 3
            forth = dataset[i]
    fclaster = 0
    sclaster = 0
    tclaster = 0
    fhclaster = 0
    fcenter = 0
    scenter = 0
    tcenter = 0
    fhcenter = 0
    flag1 = 1
    flag2 = 1
    flag3 = 1
    flag4 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            tdist = dist(third, dataset[i])
            fhdist = dist(forth, dataset[i])
            mindist = min(fdist, sdist, tdist, fhdist)
            if fdist == mindist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif sdist == mindist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            elif tdist == mindist:
                if flag3:
                    tclaster = np.array(dataset[i])
                    flag3 = 0
                else:
                    tclaster = np.vstack((tclaster, dataset[i]))
            else:
                if flag4:
                    fhclaster = np.array(dataset[i])
                    flag4 = 0
                else:
                    fhclaster = np.vstack((fhclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        for i in np.arange(tclaster.shape[0]):
            tcenter = tcenter + tclaster[i]
        tcenter = tcenter / tclaster.shape[0]
        for i in np.arange(fhclaster.shape[0]):
            fhcenter = fhcenter + fhclaster[i]
        fhcenter = fhcenter / fhclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps) and (dist(third, tcenter) < eps) and (dist(forth, fhcenter) < eps):
            break
        first = fcenter
        second = scenter
        third = tcenter
        forth = fhcenter
        flag1 = 1
        flag2 = 1
        flag3 = 1
        flag4 = 1
        fclaster = 0
        sclaster = 0
        tclaster = 0
        fhclaster = 0
        fcenter = 0
        scenter = 0
        tcenter = 0
        fhcenter = 0

    fig2 = plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    xpoints, ypoints = np.hsplit(fclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'red', s = 1)
    xpoints, ypoints = np.hsplit(sclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
    xpoints, ypoints = np.hsplit(tclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'green', s = 1)
    xpoints, ypoints = np.hsplit(fhclaster, 2)
    plt.scatter(xpoints, ypoints, c = 'yellow', s = 1)
    plt.savefig('clastered_4.png', dpi = 800, format = 'png')

if s == 'auto':
    size = dataset.shape
    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    fclaster = 0
    sclaster = 0
    fcenter = 0
    scenter = 0
    flag1 = 1
    flag2 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            if fdist < sdist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif fdist > sdist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            else:
                if random.randint(0, 1):
                    if flag2:
                        sclaster = np.array(dataset[i])
                        flag2 = 0
                    else:
                        sclaster = np.vstack((sclaster, dataset[i]))
                else:
                    if flag1:
                        fclaster = np.array(dataset[i])
                        flag1 = 0
                    else:
                        fclaster = np.vstack((fclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps):
            break
        first = fcenter
        second = scenter
        flag1 = 1
        flag2 = 1
        fclaster = 0
        sclaster = 0
        fcenter = 0
        scenter = 0
    fclastercp1 = fclaster.copy()
    sclastercp1 = sclaster.copy()
    fclasdist = 0
    sclasdist = 0
    for i in np.arange(fclaster.shape[0]):
        fclasdist = fclasdist + dist(fcenter, fclaster[i])
    fclasdist = fclasdist / fclaster.shape[0]
    for i in np.arange(sclaster.shape[0]):
        sclasdist = sclasdist + dist(scenter, sclaster[i])
    sclasdist = sclasdist / sclaster.shape[0]
    twoclas = math.sqrt((fclasdist ** 2 + sclasdist ** 2) / 2)

    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i])) / 2) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i])) / 2
            third = dataset[i]
    fclaster = 0
    sclaster = 0
    tclaster = 0
    fcenter = 0
    scenter = 0
    tcenter = 0
    flag1 = 1
    flag2 = 1
    flag3 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            tdist = dist(third, dataset[i])
            mindist = min(fdist, sdist, tdist)
            if fdist == mindist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif sdist == mindist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            else:
                if flag3:
                    tclaster = np.array(dataset[i])
                    flag3 = 0
                else:
                    tclaster = np.vstack((tclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        for i in np.arange(tclaster.shape[0]):
            tcenter = tcenter + tclaster[i]
        tcenter = tcenter / tclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps) and (dist(third, tcenter) < eps):
            break
        first = fcenter
        second = scenter
        third = tcenter
        flag1 = 1
        flag2 = 1
        flag3 = 1
        fclaster = 0
        sclaster = 0
        tclaster = 0
        fcenter = 0
        scenter = 0
        tcenter = 0
    fclastercp2 = fclaster.copy()
    sclastercp2 = sclaster.copy()
    tclastercp2 = tclaster.copy()
    fclasdist = 0
    sclasdist = 0
    tclasdist = 0
    for i in np.arange(fclaster.shape[0]):
        fclasdist = fclasdist + dist(fcenter, fclaster[i])
    fclasdist = fclasdist / fclaster.shape[0]
    for i in np.arange(sclaster.shape[0]):
        sclasdist = sclasdist + dist(scenter, sclaster[i])
    sclasdist = sclasdist / sclaster.shape[0]
    for i in np.arange(tclaster.shape[0]):
        tclasdist = tclasdist + dist(tcenter, tclaster[i])
    tclasdist = tclasdist / tclaster.shape[0]
    threeclas = math.sqrt((fclasdist ** 2 + sclasdist ** 2 + tclasdist ** 2) / 3)

    first = dataset[random.randint(0, size[0] - 1)]
    maxdist = 0
    for i in np.arange(size[0]):
        if dist(first, dataset[i]) > maxdist:
            maxdist = dist(first, dataset[i])
            second = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i])) / 2) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i])) / 2
            third = dataset[i]
    maxdist = 0
    for i in np.arange(size[0]):
        if ((dist(first, dataset[i]) + dist(second, dataset[i]) + dist(third, dataset[i])) / 3) > maxdist:
            maxdist = (dist(first, dataset[i]) + dist(second, dataset[i]) + dist(third, dataset[i])) / 3
            forth = dataset[i]
    fclaster = 0
    sclaster = 0
    tclaster = 0
    fhclaster = 0
    fcenter = 0
    scenter = 0
    tcenter = 0
    fhcenter = 0
    flag1 = 1
    flag2 = 1
    flag3 = 1
    flag4 = 1
    while True:
        for i in np.arange(size[0]):
            fdist = dist(first, dataset[i])
            sdist = dist(second, dataset[i])
            tdist = dist(third, dataset[i])
            fhdist = dist(forth, dataset[i])
            mindist = min(fdist, sdist, tdist, fhdist)
            if fdist == mindist:
                if flag1:
                    fclaster = np.array(dataset[i])
                    flag1 = 0
                else:
                    fclaster = np.vstack((fclaster, dataset[i]))
            elif sdist == mindist:
                if flag2:
                    sclaster = np.array(dataset[i])
                    flag2 = 0
                else:
                    sclaster = np.vstack((sclaster, dataset[i]))
            elif tdist == mindist:
                if flag3:
                    tclaster = np.array(dataset[i])
                    flag3 = 0
                else:
                    tclaster = np.vstack((tclaster, dataset[i]))
            else:
                if flag4:
                    fhclaster = np.array(dataset[i])
                    flag4 = 0
                else:
                    fhclaster = np.vstack((fhclaster, dataset[i]))
        for i in np.arange(fclaster.shape[0]):
            fcenter = fcenter + fclaster[i]
        fcenter = fcenter / fclaster.shape[0]
        for i in np.arange(sclaster.shape[0]):
            scenter = scenter + sclaster[i]
        scenter = scenter / sclaster.shape[0]
        for i in np.arange(tclaster.shape[0]):
            tcenter = tcenter + tclaster[i]
        tcenter = tcenter / tclaster.shape[0]
        for i in np.arange(fhclaster.shape[0]):
            fhcenter = fhcenter + fhclaster[i]
        fhcenter = fhcenter / fhclaster.shape[0]
        if (dist(first, fcenter) < eps) and (dist(second, scenter) < eps) and (dist(third, tcenter) < eps) and (dist(forth, fhcenter) < eps):
            break
        first = fcenter
        second = scenter
        third = tcenter
        forth = fhcenter
        flag1 = 1
        flag2 = 1
        flag3 = 1
        flag4 = 1
        fclaster = 0
        sclaster = 0
        tclaster = 0
        fhclaster = 0
        fcenter = 0
        scenter = 0
        tcenter = 0
        fhcenter = 0
    fclasdist = 0
    sclasdist = 0
    tclasdist = 0
    fhclasdist = 0
    for i in np.arange(fclaster.shape[0]):
        fclasdist = fclasdist + dist(fcenter, fclaster[i])
    fclasdist = fclasdist / fclaster.shape[0]
    for i in np.arange(sclaster.shape[0]):
        sclasdist = sclasdist + dist(scenter, sclaster[i])
    sclasdist = sclasdist / sclaster.shape[0]
    for i in np.arange(tclaster.shape[0]):
        tclasdist = tclasdist + dist(tcenter, tclaster[i])
    tclasdist = tclasdist / tclaster.shape[0]
    for i in np.arange(fhclaster.shape[0]):
        fhclasdist = fhclasdist + dist(fhcenter, fhclaster[i])
    fhclasdist = fhclasdist / fhclaster.shape[0]
    fourclas = math.sqrt((fclasdist ** 2 + sclasdist ** 2 + tclasdist ** 2 + fhclasdist ** 2) / 4)
    bestclas = min(twoclas, threeclas, fourclas)
    if twoclas == bestclas:
        fig2 = plt.figure()
        plt.xlabel('X')
        plt.ylabel('Y')
        xpoints, ypoints = np.hsplit(fclastercp1, 2)
        plt.scatter(xpoints, ypoints, c = 'red', s = 1)
        xpoints, ypoints = np.hsplit(sclastercp1, 2)
        plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
        plt.savefig('clastered_2.png', dpi = 800, format = 'png')
    elif threeclas == bestclas:
        fig2 = plt.figure()
        plt.xlabel('X')
        plt.ylabel('Y')
        xpoints, ypoints = np.hsplit(fclastercp2, 2)
        plt.scatter(xpoints, ypoints, c = 'red', s = 1)
        xpoints, ypoints = np.hsplit(sclastercp2, 2)
        plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
        xpoints, ypoints = np.hsplit(tclastercp2, 2)
        plt.scatter(xpoints, ypoints, c = 'green', s = 1)
        plt.savefig('clastered_3.png', dpi = 800, format = 'png')
    else:
        fig2 = plt.figure()
        plt.xlabel('X')
        plt.ylabel('Y')
        xpoints, ypoints = np.hsplit(fclaster, 2)
        plt.scatter(xpoints, ypoints, c = 'red', s = 1)
        xpoints, ypoints = np.hsplit(sclaster, 2)
        plt.scatter(xpoints, ypoints, c = 'blue', s = 1)
        xpoints, ypoints = np.hsplit(tclaster, 2)
        plt.scatter(xpoints, ypoints, c = 'green', s = 1)
        xpoints, ypoints = np.hsplit(fhclaster, 2)
        plt.scatter(xpoints, ypoints, c = 'yellow', s = 1)
        plt.savefig('clastered_4.png', dpi = 800, format = 'png')
