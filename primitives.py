import taichi as ti
import taichi.math as tm


# Functions
@ti.func
def faceforward(dir, nor):
    # Correct The Normal Vector Which Is Same For Both Sides
    return -tm.sign(tm.dot(dir, nor)) * nor

@ti.func
def rotate(vector, rotation):
    # Rotation Of Vector For Given Angle
    rotation = tm.pi * rotation / 180.0
    rotatedvector = tm.rotation3d(rotation[0], rotation[1], rotation[2]) @ tm.vec4(vector, 1.0)
    return tm.vec3(rotatedvector[0], rotatedvector[1], rotatedvector[2])


# Material
@ti.dataclass
class material:
    reflection: tm.vec3
    absorption: tm.vec4
    scattering: tm.vec4
    coat: tm.vec3
    emission: tm.vec3
    refractiveindex: tm.vec3
    roughness: float
    reflectance: float
    coatroughness: float
    coatintensity: float
    transmittance: float
    luminosity: float


# 3D Objects Data
@ti.dataclass
class objects:
    pos: tm.vec3
    var: tm.vec3
    rotation: tm.vec3
    id: int

class ellipsoid:
    def __init__(self, pos, var, rotation):
        self.pos = pos
        self.var = var
        self.rotation = rotation
    
    @ti.func
    def intersect(self, origin, dir):
        radinvsq = tm.pow(self.var, -2.0)
        relorg = rotate(origin - self.pos, self.rotation)
        rotdir = rotate(dir, self.rotation)
        a = tm.dot(rotdir * radinvsq, rotdir)
        b = 2 * tm.dot(relorg * radinvsq, rotdir)
        c = tm.dot(relorg * radinvsq, relorg) - 1
        discriminant = b*b - 4*a*c
        t1 = (-b - tm.sqrt(discriminant)) / (2*a)
        t2 = (-b + tm.sqrt(discriminant)) / (2*a)
        t = ti.select(discriminant < 0, float('inf'), t1)
        inside = ti.select(t < 0, 1.0, -1.0)
        t = ti.select(t < 0, t2, t)
        inside = ti.select(t > 1e-4, inside, -1.0)
        t = ti.select(t > 1e-4, t, float('inf'))
        return t, rotate(tm.normalize((relorg + rotdir*t) * radinvsq) * -inside, -self.rotation), inside
    
    @ti.func
    def sdf(self, p):
        rel_p = rotate(p - self.pos, self.rotation)
        return tm.length(rel_p / self.var) - 1.0
    
    @ti.func
    def id(self):
        return 1

class plane:
    def __init__(self, pos, var, rotation):
        self.pos = pos
        self.var = var
        self.rotation = rotation

    @ti.func
    def intersect(self, origin, dir):
        relorg = rotate(origin - self.pos, self.rotation)
        dir = rotate(dir, self.rotation)
        t = -tm.dot(self.var, relorg) / tm.dot(self.var, dir)
        t = ti.select(t > 1e-4, t, float('inf'))
        return t, rotate(faceforward(dir, self.var), -self.rotation), -1.0
    
    @ti.func
    def sdf(self, p):
        rel_p = rotate(p - self.pos, self.rotation)
        return rel_p[1]
    
    @ti.func
    def id(self):
        return 2

class box:
    def __init__(self, pos, var, rotation):
        self.pos = pos
        self.var = var
        self.rotation = rotation

    # https://iquilezles.org/articles/boxfunctions/
    @ti.func
    def intersect(self, origin, dir):
        relorg = rotate(origin - self.pos, self.rotation)
        rotdir = rotate(dir, self.rotation)
        m = 1 / rotdir
        n = m * relorg
        k = tm.max(m, -m) * self.var / 2
        k1 = -n - k
        k2 = -n + k
        tN = tm.max(k1[0], k1[1], k1[2])
        tF = tm.min(k2[0], k2[1], k2[2])
        t1 = tm.min(tN, tF)
        t2 = tm.max(tN, tF)
        inside = ti.select(t1 < 0, 1.0, -1.0)
        t = ti.select(t1 < 0, t2, t1)
        inside = ti.select(tN > tF, -1.0, inside)
        t = ti.select(tN > tF, float('inf'), t)
        inside = ti.select(t > 1e-4, inside, -1.0)
        t = ti.select(t > 1e-4, t, float('inf'))
        return t, rotate(tm.normalize(-tm.sign(rotdir) * ti.select(tN > 0, tm.step(tm.vec3(tN), k1), tm.step(k2, tm.vec3(tF)))), -self.rotation), inside
    '''
    @ti.func
    def intersect(self, origin, dir):
        relorg = origin - self.var1
        size = self.var2 / 2
        #dir = ti.max(dir, -dir)
        n1 = (size / dir) - (relorg / dir)
        n2 = (-size / dir) - (relorg / dir)
        k1 = tm.min(n1, n2)
        k2 = tm.max(n1, n2)
        t1 = tm.min(tm.max(n2[0], -n2[0]), tm.max(n1[1], -n1[1]))
        t = ti.select(t1 > 0, t1, float('inf'))
        return (t, tm.vec3(0.0, 1.0, 0.0), -1.0)'''
    
    # https://iquilezles.org/articles/boxfunctions/
    @ti.func
    def sdf(self, p):
        rel_p = rotate(p - self.pos, self.rotation)
        new_size = self.var / 2.0
        d = tm.max(rel_p, -rel_p) - new_size
        return tm.length(tm.max(d, 0.0)) + tm.min(tm.max(d[0], d[1], d[2]), 0.0)
    
    @ti.func
    def id(self):
        return 3

