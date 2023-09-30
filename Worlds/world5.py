import taichi as ti
import taichi.math as tm
import primitives as ob

def worlddata(time):
    time = time * tm.pi / 4.0
    ray_origin = [0.8, 0.6, -1.5]
    theta = [0.0, 0.0]
    # Objects are written in this order: [position, radius, color, roughness, luminosity, emission color]
    ground = ob.plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], 1.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid1 = ob.ellipsoid([0.5, 0.2, 0.7], [0.5, 0.5, 0.5], [0.08, 0.9, 0], 1.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid2 = ob.ellipsoid([0, 0.35, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.9], 1.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid3 = ob.ellipsoid([-0.1, 0.0, 0.0], [0.25, 0.25, 0.25], [0.95, 0.1, 0.0], 1.0, 0.0, [0.0, 0.0, 0.0])
    light1 = ob.ellipsoid([1.7, 1.5, 0.3], [0.15, 0.15, 0.15], [0.0, 0.0, 0.0], 1.0, 100.0, [0.7, 0.9, 1.0])
    objects = [ground, ellipsoid1, ellipsoid2, ellipsoid3, light1]
    return (ray_origin, theta, objects)

@ti.func
def sky(dir):
    color = tm.vec3(0, 0, 0)
    return color
