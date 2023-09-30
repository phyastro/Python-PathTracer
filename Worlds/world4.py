import taichi as ti
import taichi.math as tm
import primitives as ob

def worlddata(time):
    time = time * tm.pi / 4.0
    ray_origin = [2.5 * tm.sin(time - tm.pi / 2) + 1.0, 0.3 + tm.sin(time) / 3, 2.5 * tm.cos(time - tm.pi / 2) + 2.0]
    theta = [time / tm.pi * 180.0, -15.0]
    # Objects are written in this order: [position, radius, color, roughness, luminosity, emission color]
    ground = ob.plane([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.6, 1.0], 1.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid1 = ob.ellipsoid([1.0, -0.25 * tm.cos(time / 2), 2.0], [1.0, 0.75, 1.0], [1.0, 0.3, 0.6], 1.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid2 = ob.ellipsoid([-1.3, -0.5, 1.0], [0.5, 0.4, 0.5], [1.0, 1.0, 1.0], 0.1, 0.0, [0.0, 0.0, 0.0])
    ellipsoid3 = ob.ellipsoid([2.0, -0.6, 0.3], [0.5, 0.6, 0.5], [1.0, 1.0, 1.0], 0.1, 0.0, [0.0, 0.0, 0.0])
    light = ob.ellipsoid([0.0, 2.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 1.0, 5.0, [1.0, 0.8, 0.9])
    objects = [ground, ellipsoid1, ellipsoid2, ellipsoid3, light]
    return (ray_origin, theta, objects)

@ti.func
def sky(dir):
    color = tm.vec3(0, 0, 0)
    return color
