import sys
import numpy as np

class Net:
    def __init__(self, fanout, netname, driver, sinks_list, sinks_type, ids, port):
        self.fanout = fanout
        self.netname = netname
        self.driver = driver
        self.sinks_list = sinks_list
        self.sinks_type = sinks_type
        self.ids = ids
        self.port = port

        self.label = self.getLabel()

    def printing(self):
        print (self.ids, self.netname, self.driver.cname, self.driver.pname, 
                         len(self.sinks_list), 'port='+str(self.port))
        #for sink in self.sinks_list:
        #    print (sink.cname)

    def getLabel(self):
        if self.driver.cname:
            Xmin, Xmax, Ymin, Ymax = self.driver.x, self.driver.x, self.driver.y, self.driver.y 
        else:
            Xmin, Xmax, Ymin, Ymax = sys.maxsize, -sys.maxsize, sys.maxsize, -sys.maxsize

        for sink in self.sinks_list:
            x, y = sink.x, sink.y
            if x > Xmax:
                Xmax = x
            if x < Xmin:
                Xmin = x
            if y > Ymax:
                Ymax = y
            if y < Ymin:
                Ymin = y 
        return Ymax - Ymin + Xmax - Xmin


class Cell:
    #def __init__(self, cname, ctype, x, y):
    def __init__(self, pname, ctype, x, y):
        self.pname = pname
        if pname:
            self.cname = pname.rsplit('/', 1)[0]
        else:
            self.cname = None
        self.ctype = ctype
        self.x = x 
        self.y = y 


def parseRaw(fn):
    with open(fn) as fp:
        nets_list = []
        finish = False
        port = 0
#        append_flag = True
        #non_driver_removed = 0
        for line in fp:
            line = line.split()

            if len(line) == 0:
                if finish:
                    if len(sinks_cname)*2 - len(sinks_X) - len(sinks_Y) != 0:
                        print ('Sink size mismatch error!')
                        exit()

                    sinks_list = []
                    for i in range(len(sinks_cname)):
                        if sinks_cname[i] != 'NotFound':
                            sink = Cell(sinks_cname[i], None, float(sinks_X[i]), float(sinks_Y[i]))
                            sinks_list.append(sink)

#                    if append_flag:
                    nets_list.append(Net(fanout, netname, driver, sinks_list, sinks_type, len(nets_list), port) )
                    #nets_list[-1].printing()
                    finish = False
                    port = 0
                    append_flag = True
            else:
                if line[0] == 'fanout:':
                    fanout = int(line[1])

                if line[0] == 'net' and line[1] == 'name:':
                    netname = str(line[2])

                if line[0] == 'driver:':
                    if line[1] != 'NotFound':
                        driver = Cell(line[1], line[2].split('/')[1], 
                                      float(line[3]), float(line[4]))
                    else:
                        driver = Cell(None, None, None, None)
                        #append_flag = False
                        #non_driver_removed += 1

                if line[0] == 'sinks:':
                    sinks_cname = line[1:]

                if line[0] == 'sink' and line[1] == 'libs:':
                    sinks_type = line[1:]

                if line[0] == 'X:':
                    sinks_X = line[1:]

                if line[0] == 'Y:':
                    sinks_Y = line[1:]
                    finish = True 

                if line[0] == 'PortsIn!!!!':
                    port = -1
                if line[0] == 'PortsOut!!!!':
                    port = 1

    print ('nets_list', len(nets_list))
    return nets_list


def buildGraph(nets_list): 
    dri_net, adj_list = dict(), dict()

    for i, n in enumerate(nets_list):
        if n.driver.cname:
            # two nets with the same driver cell
            if n.driver.cname not in dri_net:
                dri_net[n.driver.cname] = [i]
            else:
                dri_net[n.driver.cname].append(i)

    for i, n in enumerate(nets_list):
        if i != n.ids:
            sys.exit('Error: id mismatch !!!')

        adj_list[i] = set()
        for sink in n.sinks_list:
            if sink.cname:
                if sink.cname not in dri_net:
                    #print ('warning in parse_net/buildGraph, sink', sink.cname, 'is not driver')
                    pass

                else:
                    for k in dri_net[sink.cname]:
                        adj_list[i].add(k)
    return adj_list


def nameIdDict(nets_list):
    id_dict = dict()
    id_sinkDict = dict()
    net_sinkDict = dict()

    for idx in range(len(nets_list)):
        n = nets_list[idx]
        if len(id_dict) != n.ids or len(id_dict) != idx:
            sys.exit('Error: id mismatch in id_dict!!')
        id_dict[n.netname] = n.ids
        #print (n.ids, n.netname, n.driver.pname)

        for sink in n.sinks_list:
            if sink.pname in id_sinkDict:
                sys.exit('Error: id mismatch in id_sinkDict!!')
            if sink.pname in net_sinkDict:
                sys.exit('Error: id mismatch in id_sinkDict!!')
            id_sinkDict[sink.pname] = n.ids
            net_sinkDict[sink.pname] = n.netname
        #    print (sink.pname)
        #print ()
    return id_dict, id_sinkDict, net_sinkDict


def mainParse(fn):
    nets_list = parseRaw(fn)
    adj_list = buildGraph(nets_list)
    # none-driver net is in remove_list
    return (nets_list, adj_list)


def generateFeaturesLabels(nets_list, area_file, fanin_file):
    area_library = readLib(area_file)
    fanin_library = readLib(fanin_file)

    features, labels = [], []
    for i, n in enumerate(nets_list):
        if n.driver.ctype and n.driver.ctype != 'inPort':
            #print (n.driver.ctype, n.driver.ctype.rsplit('X', 1)[0])
            features.append([area_library[n.driver.ctype], fanin_library[n.driver.ctype]])
        else:
            #print (i, n.netname, n.driver.ctype, n.label)
            features.append([0, 0])

        labels.append(n.label)
    return (np.array(features), np.array(labels))

def readLib(lib_file):
    library = {}
    with open(lib_file,"r") as in_file:
        for line in in_file:
            cell_pin, cap = line.split(",")
            library[cell_pin] = float(cap)
    return library



def reverseGraph(adj_list):
    rev_list = dict()
    for i in range(len(adj_list)):
        rev_list[i] = set()

    for k, v in adj_list.items():
        for vi in v:
            rev_list[vi].add(k)
    return rev_list


def mergeGraph(adj_list, rev_list):
    for i in range(len(adj_list)):
    #    print ('adj, rev, adj+rev',
    #           len(adj_list[i]), len(rev_list[i]), len(adj_list[i].union(rev_list[i])))
        adj_list[i] = adj_list[i].union(rev_list[i])
    return adj_list


