from Tkinter import *
import random

DO_EDGES = True

WIDTH = 1000.0
HEIGHT = WIDTH
point_count = 500

min_points_per_node = 2
min_edges_per_node = 1
min_size_of_node = 12

p_1 = [float(random.randint(1, WIDTH-1)),float(random.randint(1, HEIGHT-1))]#[800.0, 200.0]
p_0 = [float(random.randint(1, WIDTH-1)),float(random.randint(1, HEIGHT-1))]#[150.0, 940.0]

point_draw_size = 6

def normalize(a):
    abssum = 0.0
    result = []
    for i in range(0,len(a)):
        abssum = abssum + abs(a[i])
    for i in range(0,len(a)):
        result.append(a[i] / abssum)
    return result

def add(a,b):
    result = []
    for i in range(0, len(a)):
        result.append(a[i] + b[i])
    return result

def sub(a,b):
    result = []
    for i in range(0, len(a)):
        result.append(a[i] - b[i])
    return result

def scale(a, s):
    result = []
    for i in range(0, len(a)):
        result.append(a[i] * s)
    return result






def onSegment(p, q, r):
    if (q[0] <= max (p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0 : return 0 #co-linear
    if val > 0 : return 1 #clockwise
    return 2 # anti-clockwise

def doIntersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2 and o3 != o4): return True

    if (o1 == 0 and onSegment(p1, p2, q1)): return True
    if (o2 == 0 and onSegment(p1, p2, q1)): return True
    if (o3 == 0 and onSegment(p2, p1, q2)): return True
    if (o4 == 0 and onSegment(p2, p1, q2)): return True

    return False




def withinBox(p, bl, tr):
    if p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1] : return True
    return False

def getNodeAtPoint(p, qt, node_list):
    indx = qt.indx
    if (qt.isLeaf): return indx
    for i in range(0,4):
        if (withinBox(p, node_list[qt.leafs[i]].bl, node_list[qt.leafs[i]].tr)):
            indx = getNodeAtPoint(p, node_list[qt.leafs[i]], node_list)
    return indx





def doesLineIntersectNode(p_0, p_1, qt_indx, node_list):
    if doIntersect(p_0, p_1, node_list[qt_indx].bl, [node_list[qt_indx].tr[0], node_list[qt_indx].bl[1]]): return True #Bottom

    if doIntersect(p_0, p_1, node_list[qt_indx].bl, [node_list[qt_indx].bl[0], node_list[qt_indx].tr[1]]): return True #Letf

    if doIntersect(p_0, p_1, node_list[qt_indx].tr, [node_list[qt_indx].tr[0], node_list[qt_indx].bl[1]]): return True #Right

    if doIntersect(p_0, p_1, node_list[qt_indx].tr, [node_list[qt_indx].bl[0], node_list[qt_indx].tr[1]]): return True #Top
    return False


def doesLineIntersectBox(p_0, p_1, bl, tr):
    if doIntersect(p_0, p_1, bl, [tr[0], bl[1]]): return True #Bottom

    if doIntersect(p_0, p_1, bl, [bl[0], tr[1]]): return True #Letf

    if doIntersect(p_0, p_1, tr, [tr[0], bl[1]]): return True #Right

    if doIntersect(p_0, p_1, tr, [bl[0], tr[1]]): return True #Top
    return False




def createEdgesLoop(points):
    edges = []
    for i in range(0, len(points) - 1):
        edges.append([points[i], points[i+1]])
    edges.append([points[0], points[-1]])
    return edges

def distFromPathToEdge(p_start, p_end, e_0, e_1):
    p1 = p_start
    d1 = normalize(sub(p_end, p_start))

    p2 = e_0
    d2 = normalize(sub(e_1, e_0))

    u = (d2[0] * (p1[1] - p2[1]) - d2[1] * (p1[0] - p2[0])) / (d2[1] * d1[0] - d2[0] * d1[1])
    return u






class QuadTree():

    def __init__(self, indx):
         self.indx = indx
         self.bl = []
         self.tr = []

         self.points = []
         self.edges = []
         self.leafs = []
         self.parent = None
         self.isLeaf = False

    def generateQuadTreePoints(self, _points, _bl, _tr, _parent, canvas, level, n_list):
        self.bl = _bl
        self.tr = _tr
        self.parent = _parent

        #print ""
        #print "bl : (", self.bl[0], ",", self.bl[1], ") : (", self.tr[0], ",", self.tr[1], ")"
        #print "length : ", len(_points)

        col = "white"
        if level == 0 : col = "white"
        if level == 1 : col = "gray"
        if level == 2 : col = "green"
        if level == 3 : col = "blue"
        if level == 4 : col = "yellow"
        if level == 5 : col = "cyan"
        if level == 6 : col = "magenta"
        if level == 7 : col = "white"
        canvas.create_rectangle(self.bl[0], self.bl[1], self.tr[0], self.tr[1], outline = col)

        if len(_points) <= min_points_per_node or _tr[0] - _bl[0] < min_size_of_node:
            self.points = _points
            self.isLeaf = True
            return
        else:
            half_x = (self.tr[0] - self.bl[0]) / 2
            half_y = (self.tr[1] - self.bl[1]) / 2

            t_points0 = []
            t_points1 = []
            t_points2 = []
            t_points3 = []

            for p in _points:
                if p[0] == self.bl[0] + half_x or p[1] == self.bl[1] + half_y:
                    self.points.append(p)
                    continue
                elif p[0] < self.bl[0] + half_x and p[1] < self.bl[1] + half_y: # Bottom Left Quadrant
                    t_points0.append(p)

                elif p[0] > self.bl[0] + half_x and p[1] < self.bl[1] + half_y: # Bottom Left Quadrant
                    t_points1.append(p)

                elif p[0] < self.bl[0] + half_x and p[1] > self.bl[1] + half_y: # Bottom Left Quadrant
                    t_points2.append(p)

                elif p[0] > self.bl[0] + half_x and p[1] > self.bl[1] + half_y: # Bottom Left Quadrant
                    t_points3.append(p)
                else:
                    print "Could not append point.\n"
            #print "child lengths : ", len(t_points0), " ", len(t_points1), " ", len(t_points2), " ", len(t_points3)

            next_node_indx = len(n_list)
            self.leafs = [next_node_indx, next_node_indx + 1, next_node_indx + 2, next_node_indx + 3]

            n_list.append(QuadTree(self.leafs[0]))
            n_list.append(QuadTree(self.leafs[1]))
            n_list.append(QuadTree(self.leafs[2]))
            n_list.append(QuadTree(self.leafs[3]))

            n_list[self.leafs[0]].generateQuadTreePoints(t_points0, self.bl                                     , [ self.tr[0] - half_x, self.tr[1] - half_y ], self.indx, canvas, level + 1, n_list)
            n_list[self.leafs[1]].generateQuadTreePoints(t_points1, [ self.bl[0] + half_x, self.bl[1]          ], [ self.tr[0],          self.tr[1] - half_y ], self.indx, canvas, level + 1, n_list)
            n_list[self.leafs[2]].generateQuadTreePoints(t_points2, [ self.bl[0],          self.bl[1] + half_y ], [ self.tr[0] - half_x, self.tr[1]          ], self.indx, canvas, level + 1, n_list)
            n_list[self.leafs[3]].generateQuadTreePoints(t_points3, [ self.bl[0] + half_x, self.bl[1] + half_y ], self.tr                                     , self.indx, canvas, level + 1, n_list)

    def generateQuadTreeEdges(self, _edges, _bl, _tr, _parent, canvas, level, n_list):
        self.bl = _bl
        self.tr = _tr
        self.parent = _parent

        print ""
        print "bl : (", self.bl[0], ",", self.bl[1], ") : (", self.tr[0], ",", self.tr[1], ")"
        print "length : ", len(_edges)

        col = "white"
        if level == 0 : col = "white"
        if level == 1 : col = "gray"
        if level == 2 : col = "green"
        if level == 3 : col = "blue"
        if level == 4 : col = "yellow"
        if level == 5 : col = "cyan"
        if level == 6 : col = "magenta"
        if level == 7 : col = "white"
        canvas.create_rectangle(self.bl[0], self.bl[1], self.tr[0], self.tr[1], outline = col)

        if len(_edges) <= min_edges_per_node or _tr[0] - _bl[0] < min_size_of_node:
            self.edges = _edges
            self.isLeaf = True
            return
        else:
            half_x = (self.tr[0] - self.bl[0]) / 2
            half_y = (self.tr[1] - self.bl[1]) / 2

            t_edges0 = []
            t_edges1 = []
            t_edges2 = []
            t_edges3 = []

            for e in _edges:
                if doesLineIntersectBox(e[0], e[1], self.bl                                     , [ self.tr[0] - half_x, self.tr[1] - half_y ]) or \
                   doesLineIntersectBox(e[0], e[1], [ self.bl[0] + half_x, self.bl[1] + half_y ], self.tr                                     ):
                   self.edges.append(e)
                else:
                   if   withinBox(e[0], self.bl                                     , [ self.tr[0] - half_x, self.tr[1] - half_y ]):
                       t_edges0.append(e)
                   elif withinBox(e[0], [ self.bl[0] + half_x, self.bl[1]          ], [ self.tr[0],          self.tr[1] - half_y ]):
                       t_edges1.append(e)
                   elif withinBox(e[0], [ self.bl[0],          self.bl[1] + half_y ], [ self.tr[0] - half_x, self.tr[1]          ]):
                       t_edges2.append(e)
                   elif withinBox(e[0], [ self.bl[0] + half_x, self.bl[1] + half_y ], self.tr                                     ):
                       t_edges3.append(e)
                   else:
                       print "Could not append edge.\n"

            print "child lengths : ", len(t_edges0), " ", len(t_edges1), " ", len(t_edges2), " ", len(t_edges3)

            genChildNodes = False
            if len(t_edges0) > 0: genChildNodes = True
            if len(t_edges1) > 0: genChildNodes = True
            if len(t_edges2) > 0: genChildNodes = True
            if len(t_edges3) > 0: genChildNodes = True

            if genChildNodes:
                next_node_indx = len(n_list)
                self.leafs = [next_node_indx, next_node_indx + 1, next_node_indx + 2, next_node_indx + 3]

                n_list.append(QuadTree(self.leafs[0]))
                n_list.append(QuadTree(self.leafs[1]))
                n_list.append(QuadTree(self.leafs[2]))
                n_list.append(QuadTree(self.leafs[3]))

                n_list[self.leafs[0]].generateQuadTreeEdges(t_edges0, self.bl                                     , [ self.tr[0] - half_x, self.tr[1] - half_y ], self.indx, canvas, level + 1, n_list) # Bottom Left
                n_list[self.leafs[1]].generateQuadTreeEdges(t_edges1, [ self.bl[0] + half_x, self.bl[1]          ], [ self.tr[0],          self.tr[1] - half_y ], self.indx, canvas, level + 1, n_list) # Bottom Right
                n_list[self.leafs[2]].generateQuadTreeEdges(t_edges2, [ self.bl[0],          self.bl[1] + half_y ], [ self.tr[0] - half_x, self.tr[1]          ], self.indx, canvas, level + 1, n_list) # Top Left
                n_list[self.leafs[3]].generateQuadTreeEdges(t_edges3, [ self.bl[0] + half_x, self.bl[1] + half_y ], self.tr                                     , self.indx, canvas, level + 1, n_list) # Top Right
            else :
                self.isLeaf = True







def generateRandomPoints(num_points, width, height):
    points = []
    for i in range(0,num_points):
        x = random.randint(1, width-1)
        y = random.randint(1, height-1)
        points.append([x,y])
    return points

def drawPoint(canvas, x, y, j):
    hs = point_draw_size / 2
    canvas.create_oval(x-hs,y-hs,x+hs,y+hs, fill = "red")
    if DO_EDGES: canvas.create_text(x + point_draw_size, y + point_draw_size, fill = "white", text = j)

def drawEdge(canvas, e0, e1, col):
    canvas.create_line(e0[0], e0[1], e1[0], e1[1], fill = col)

def redrawAll(canvas, points):
    canvas.delete(ALL)

    canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="black")
    if DO_EDGES:
        for i in range(0, len(points)-1):
            drawEdge(canvas, points[i], points[i+1], "white")
        drawEdge(canvas, points[0], points[-1], "white")

    j = 0
    for p in points:
        drawPoint(canvas, p[0], p[1], j)
        j = j + 1


def fillNode(indx, node_list, canvas):
    canvas.create_rectangle(node_list[indx].bl[0], node_list[indx].bl[1], node_list[indx].tr[0], node_list[indx].tr[1], fill = "red", stipple = "gray12", outline = "red")

def init(canvas):
    redrawAll(canvas)






########### copy-paste below here ###########

def run():
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=WIDTH, height=HEIGHT)


    p = generateRandomPoints(point_count, WIDTH, HEIGHT)
    if DO_EDGES: p = [[450,350],[440,460],[460,440],[530,320],[620,420],[605,440],[740,550],[780,710],[780,730],[760,730],[550,640],[470,840],[360,750],[360,840],[280,820],[290,830],[260,850],[260,780],[220,790],[230,740],[160,760],[100,820],[100,20],[125,90]]
    e = createEdgesLoop(p)




    canvas.pack()
    # Store canvas in root and in canvas itself for callbacks
    root.canvas = canvas.canvas = canvas
    # Set up canvas data and call init
    canvas.data = { }
    #init(canvas)
    redrawAll(canvas, p)




    node_list = []

    qTree = QuadTree(0)
    node_list.append(qTree)

    if DO_EDGES :
        qTree.generateQuadTreeEdges(e, [0,0], [WIDTH, HEIGHT], None, canvas, 0, node_list)
    else:
        qTree.generateQuadTreePoints(p, [0,0], [WIDTH,HEIGHT], None, canvas, 0, node_list)
    print len(node_list)


    node_intersect_list = []

    for i in range(0, len(node_list)):
        if doesLineIntersectNode(p_0, p_1, i, node_list):

            n_ind = node_list[i].indx
            node_intersect_list.append(n_ind)
            if n_ind in [1,2,3,4] and 0 not in node_intersect_list: node_intersect_list.append(0)

            if node_list[i].isLeaf: fillNode(i, node_list, canvas)
    if len(node_intersect_list) < 1:
        start = getNodeAtPoint(p_0, node_list[0], node_list)
        node_intersect_list.append(start)
        fillNode(start, node_list, canvas)

    print len(node_intersect_list)
    print node_intersect_list


    if DO_EDGES:
        edge_intersection_list = []
        edge_dist_list = []
        for i in range(0, len(node_intersect_list)):
            for e in node_list[node_intersect_list[i]].edges:
                if doIntersect(e[0], e[1], p_0, p_1):
                    d = distFromPathToEdge(p_0, p_1, e[0], e[1])

                    print "intersection"

                    edge_intersection_list.append(e)
                    edge_dist_list.append(d)



        if len(edge_intersection_list) >=1:
            smallest_indx = 0
            smallest_dist = edge_dist_list[0]

            if len(edge_intersection_list) > 1:

                for i in range(1, len(edge_intersection_list)):
                    if edge_dist_list[i] < smallest_dist:
                        smallest_indx = i
                        smallest_dist = edge_dist_list[i]

            e = edge_intersection_list[smallest_indx]
            drawEdge(canvas, e[0], e[1], "green")



    canvas.create_line(p_0[0], p_0[1], p_1[0], p_1[1], fill = "white")
    canvas.create_text(p_0[0] + 10, p_0[1] + 10, fill = "white", text = "p_0")
    canvas.create_text(p_1[0] + 10, p_1[1] + 10, fill = "white", text = "p_1")

    # set up events
    # root.bind("<Button-1>", mousePressed)
    # root.bind("<Key>", keyPressed)
    # timerFired(canvas)
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits until you close the window!)

run()
