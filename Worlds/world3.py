import taichi as ti
import taichi.math as tm
import primitives as ob
import math

def worlddata(time):
    ray_origin = [-3.75, 0.6, 0.0]
    theta = [2.0, -3.0]
    timetoangle = math.pi * time / 5.0
    # Objects are written in this order: [position, radius, color, roughness, luminosity, emission color]
    ground = ob.ellipsoid([0.0, -10002.0, 0.0], [10000.0, 10000.0, 10000.0], [0.9, 0.9, 0.9], 1.0, 0.0, [0.0, 0.0, 0.0])
    roof = ob.ellipsoid([0.0, 10002.0, 0.0], [10000.0, 10000.0, 10000.0], [0.3, 0.3, 0.9], 1.0, 0.0, [0.0, 0.0, 0.0])
    wall1 = ob.ellipsoid([10002.0, 0.0, 0.0], [10000.0, 10000.0, 10000.0], [0.4, 0.8, 0.9], 1.0, 0.0, [0.0, 0.0, 0.0])
    wall2 = ob.ellipsoid([-10004.0, 0.0, 0.0], [10000.0, 10000.0, 10000.0], [0.9, 0.2, 0.9], 1.0, 0.0, [0.0, 0.0, 0.0])
    wall3 = ob.ellipsoid([0.0, 0.0, 10002.0], [10000.0, 10000.0, 10000.0], [0.7, 0.1, 0.1], 1.0, 0.0, [0.0, 0.0, 0.0])
    wall4 = ob.ellipsoid([0.0, 0.0, -10002.0], [10000.0, 10000.0, 10000.0], [0.1, 0.9, 0.1], 1.0, 0.0, [0.0, 0.0, 0.0])
    light1 = ob.ellipsoid([1.6 * tm.sin(timetoangle), 1.6 * tm.cos(timetoangle), 1.6 * tm.cos(timetoangle)], [0.4, 0.4, 0.4], [0.0, 0.0, 0.0], 1.0, 1.0, [0.05, 1.0, 0.05])
    light2 = ob.ellipsoid([1.6 * tm.sin(timetoangle), 1.6 * tm.cos(timetoangle), -1.6 * tm.cos(timetoangle)], [0.4, 0.4, 0.4], [0.0, 0.0, 0.0], 1.0, 1.0, [0.07, 0.07, 1.0])
    light3 = ob.ellipsoid([-1.6 * tm.sin(timetoangle), -1.6 * tm.cos(timetoangle), -1.6 * tm.cos(timetoangle)], [0.4, 0.4, 0.4], [0.0, 0.0, 0.0], 1.0, 1.0, [1.0, 1.0, 1.0])
    light4 = ob.ellipsoid([-1.6 * tm.sin(timetoangle), -1.6 * tm.cos(timetoangle), 1.6 * tm.cos(timetoangle)], [0.4, 0.4, 0.4], [0.0, 0.0, 0.0], 1.0, 1.0, [1.0, 0.0, 0.0])
    ellipsoid1 = ob.ellipsoid([0.0, 0.0, 1.6], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0], 0.0, 0.0, [0.0, 0.0, 0.0])
    ellipsoid2 = ob.ellipsoid([0.0, 0.0, 0.8], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0], 0.25, 0.0, [0.0, 0.0, 0.0])
    ellipsoid3 = ob.ellipsoid([0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0], 0.5, 0.0, [0.0, 0.0, 0.0])
    ellipsoid4 = ob.ellipsoid([0.0, 0.0, -0.8], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0], 0.75, 0.0, [0.0, 0.0, 0.0])
    ellipsoid5 = ob.ellipsoid([0.0, 0.0, -1.6], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0], 1.0, 0.0, [0.0, 0.0, 0.0])
    objects = [ground, roof, wall1, wall2, wall3, wall4, light1, light2, light3, light4, ellipsoid1, ellipsoid2, ellipsoid3, ellipsoid4, ellipsoid5]
    return (ray_origin, theta, objects)

@ti.func
def sky(dir):
    color = tm.vec3(0, 0, 0)
    return color
