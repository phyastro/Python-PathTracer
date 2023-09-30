import taichi as ti
import taichi.math as tm
import primitives as ob

# https://iquilezles.org/articles/smin/
@ti.func
def smin(a, b, k):
    h = tm.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return tm.mix(b, a, h) - k * h * (1.0 - h)

@ti.func
def SpectralPowerDistribution(l, l_peak, d, invert):
    # Spectral Power Distribution Function Calculated On The Basis Of Peak Wavelength And Standard Deviation
    # Using Gaussian Function To Predict Spectral Radiance
    # In Reality, Spectral Radiance Function Has Different Shapes For Different Objects Also Looks Much Different Than This
    radiance = tm.exp(-((l - l_peak) / (2 * d * d)) ** 2)
    radiance = tm.mix(radiance, 1.0 - radiance, invert)
    return radiance

def Camera(time):
    ray_origin = [6.332, 3.855, 3.140]
    #ray_origin = [-2.063, 3.724, 4.332]
    #ray_origin = [0.72, 3.724, 4.943]
    #ray_origin = [6.511, 5.354, -6.399]
    theta = [135.093, -31.512]
    #theta = [57.744, -26.737]
    #theta =  [88.63, -28.169]
    #theta = [224.26, -36.287]
    lensradius = 0.5
    focal_length = 5.25
    lensroughness = 0.0
    lensrefractindex = tm.vec3(1.010, 550.0, 0.025)
    return (ray_origin, theta, (lensradius, focal_length), lensroughness, lensrefractindex)

def Intersectors(time):
    # Materials Are Written In This Way:
    # [reflection[wavelength, deviation, invert], absorption[wavelength, deviation, invert, absorptivity], scattering[wavelength, deviation, invert, scattering density], coat[wavelength, deviation, invert], emission[wavelength, deviation, invert], refractiveindex[refractiveindex at wavelength n, wavelength n, curve sloppiness], roughness, reflectance, coatroughness, coatintensity, transmittance, luminosity]
    groundmaterial = ob.material([550.0, 100.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    glassmaterial = ob.material([550.0, 100.0, 0], [515.0, 7.5, 1, 1.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.52, 300.0, 0.06], 0.03, 0.5, 1.0, 0.0, 0.96, 0.0)
    coatedbluematerial = ob.material([470.0, 6.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 100.0, 0], [550.0, 0.0, 0], [1.25, 550.0, 0.0], 1.0, 0.0, 0.05, 0.13, 0.0, 0.0)
    goldmaterial = ob.material([590.0, 7.3, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.7, 350.0, 1.15], 0.01, 0.5, 0.0, 0.0, 0.0, 0.0)
    #volumematerial = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [450.0, 7.0, 0, 1.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.0, 550.0, 0.0], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    whitelightmaterial = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [500.0, 11.0, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 0.38)
    #redlightmaterial = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [650.0, 2.5, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 2.06)
    #greenlightmaterial = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 2.5, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 2.06)
    #bluelightmaterial = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [450.0, 2.5, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 2.06)
    #sunmaterial1 = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [600.0, 11.0, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 20.59)
    #sunmaterial2 = ob.material([550.0, 0.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [600.0, 11.0, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 0.09)
    # Objects Are Written In This Way:
    # (ob.object(tm.vec3(position), tm.vec3(radius/normal/size), tm.vec3(rotation)), material)
    ground = ['plane', tm.vec3(0.0, 0.0, 0.0), tm.vec3(0.0, 1.0, 0.0), tm.vec3(0.0, 0.0, 0.0), groundmaterial]
    glasssphere = ['sphere', tm.vec3(0.0, 1.0, 0.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0), glassmaterial]
    glasscube = ['box', tm.vec3(3.0, 0.75, 1.0), tm.vec3(1.5, 1.5, 1.5), tm.vec3(0.0, 0.0, 58.31), glassmaterial]
    coatedbluesphere = ['sphere', tm.vec3(5.0, 1.0, -1.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0), coatedbluematerial]
    goldcube = ['box', tm.vec3(4.0, 1.0, -4.0), tm.vec3(2.0, 2.0, 2.0), tm.vec3(0.0, 0.0, 0.0), goldmaterial]
    glasssheet = ['box', tm.vec3(-2.5, 1.5, -1.5), tm.vec3(0.05, 3.0, 4.0), tm.vec3(0.0, 0.0, 0.0), glassmaterial]
    #volumebox = ['box', tm.vec3(0.0, 2.0, -6.0), tm.vec3(2.0, 2.0, 2.0), tm.vec3(0.0, 0.0, 0.0), volumematerial]
    sphericallight = ['sphere', tm.vec3(0.0, 4.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0), whitelightmaterial]
    #redcubelight = (ob.box(tm.vec3(-2.0, 4.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0)), redlightmaterial)
    #greencubelight = (ob.box(tm.vec3(0.0, 4.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0)), greenlightmaterial)
    #bluecubelight = (ob.box(tm.vec3(2.0, 4.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0)), bluelightmaterial)
    #sun1 = (ob.ellipsoid(tm.vec3(0.0, 2000.0, -4000.0), tm.vec3(140.0, 140.0, 140.0), tm.vec3(0.0, 0.0, 0.0)), sunmaterial1)
    #sun2 = (ob.ellipsoid(tm.vec3(0.0, 4000.0, -4000.0), tm.vec3(4000.0, 4000.0, 4000.0), tm.vec3(0.0, 0.0, 0.0)), sunmaterial2)
    objects = [ground, glasssphere, glasscube, coatedbluesphere, goldcube, glasssheet, sphericallight]
    return objects

@ti.func
def Raymarchers(p, time):
    # Materials Are Written In This Way:
    # [reflection[wavelength, deviation, invert], absorption[wavelength, deviation, invert, absorptivity], scattering[wavelength, deviation, invert, scattering density], coat[wavelength, deviation, invert], emission[wavelength, deviation, invert], refractiveindex[refractiveindex at wavelength n, wavelength n, curve sloppiness], roughness, coatroughness, coatintensity, transmittance, luminosity]
    orangediffusematerial = ob.material([660.0, 6.0, 0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.0, 550.0, 0.0], 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    #glassmaterial = ob.material([550.0, 100.0, 0], [515.0, 7.5, 1, 1.0], [550.0, 0.0, 0, 0.0], [550.0, 0.0, 0], [550.0, 0.0, 0], [1.52, 300.0, 0.06], 0.0, 0.5, 1.0, 0.0, 0.96, 0.0)
    # Objects Are Written In This Way:
    # (ob.object.sdf(p, tm.vec3(position), tm.vec3(radius/normal/size), tm.vec3(rotation)), material)
    sphere1 = ob.ellipsoid(tm.vec3(-0.7, 1.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0))
    sphere2 = ob.ellipsoid(tm.vec3(1.3, 1.0, -3.0), tm.vec3(1.0, 1.0, 1.0), tm.vec3(0.0, 0.0, 0.0))
    boundingsphere = ob.ellipsoid(((sphere1.pos + sphere2.pos) / 2.0), tm.vec3(tm.length(sphere2.pos - sphere1.pos)), tm.vec3(0.0, 0.0, 0.0))
    orangediffuseobject = (smin(sphere1.sdf(p), sphere2.sdf(p), 0.4), boundingsphere, orangediffusematerial)
    objects = [orangediffuseobject]
    return objects

@ti.func
def Sky(dir, l):
    #radiance = SpectralPowerDistribution(tm.max(dir[1], 0.001) * l, 180.0, 12.0, 0) * 0.015
    radiance = 0.0
    return radiance
