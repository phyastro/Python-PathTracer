import taichi as ti
import taichi.math as tm
import warnings
import os
import math
from tqdm import tqdm
import time
from tkinter import filedialog
import importlib.util


# Settings
warnings.filterwarnings("ignore")
ti.init(arch=ti.vulkan, offline_cache=True)
default = bool(int(input('Use Default Settings?: ')))
if default:
    image_height = 360
    image_width = 640
    fov = 60
    step_paths = 5
    path_length = 10
    depth_of_field = False
    Renderimage = False
else:
    image_height = int(input('Height of the image(default 360):'))
    image_width = int(input('Width of the image(default 640):'))
    fov = float(input('Field of view(default 60):'))
    step_paths = int(input('Number of paths for each step:'))
    path_length = int(input('Length of the path:'))
    depth_of_field = bool(int(input('Depth of field?:')))
    Renderimage = bool(int(input('Render(1) or realtime preview(0):')))
if Renderimage == True:
    num_paths = int(input('Total number of paths of a frame:'))
    animation = bool(int(input('Animation(1) or image(0):')))
    if animation == True:
        fps_0 = int(input('Frames per second of the animation:'))
        frames_start = int(input('Start frame number:'))
        frames_end = int(input('End frame number:'))
    else:
        ui = bool(int(input('Live view of rendering?:')))
        time_s = float(input('Time at which scene has to be rendered:'))
else:
    num_paths = 0
MIS = True


# Import Includes
import primitives as ob
import spectrum


# Open World File
worldfilepath = filedialog.askopenfile(title="Select World File", filetypes=[("Python files", "*.py")])


#Ask Save File Path
if Renderimage == True:
    if animation == True:
        animdirectory = filedialog.askdirectory(title="Animation Output Directory")
    else:
        Savedirectory = filedialog.asksaveasfilename(title="Save Image As", filetypes=[("Portable Network Graphics", "*.png")])


# Precomputed Data
aspect_ratio = image_width / image_height
tan_fov = math.tan(math.radians(fov / 2.0))


# Create Fields
pixels = ti.Vector.field(n=3, dtype=ti.f64, shape=(image_width, image_height))
imagepixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(image_width, image_height))


# Tonemap Stuff
@ti.func
def Gamma(x, y):
    """Gamma Function (x^(1/y))"""
    x = ti.select(x < 0, 0, x)
    x = ti.select(x > 1, 1, x)
    return ti.cast(x, ti.f32) ** (1 / y)

@ti.func
def BioPhotometricTonemap(x):
    """Biophotometric Tonemapping by Ted"""
    c = 0.33
    p = 1.10
    b = 3.50
    return (-(c / ((ti.cast(x, ti.f32) ** p) + c)) + 1.0) ** b


# Functions
@ti.func
def SampleSpectral(start, end, input):
    """Uniform Inverted CDF For Sampling"""
    return (end - start) * input + start

@ti.func
def SpectralPDF(start, end):
    """Uniform PDF"""
    return 1.0 / (end - start)

@ti.func
def HeroSpectralSampling(l_h, j, l_min, l_max, C):
    """Hero Wavelength Spectral sampling"""
    l_d = l_max - l_min
    return tm.mod((l_h - l_min) + ((j / C) * (l_d)), l_d) + l_min

@ti.func
def ObjectIntersect(pos, var, rotation, id:ti.int32, origin, dir):
    hitdist, normal, inside = float('inf'), tm.vec3(0.0, 0.0, 0.0), -1.0
    if id == 1:
        hitdist, normal, inside = ob.ellipsoid(pos, var, rotation).intersect(origin, dir)
    elif id == 2:
        hitdist, normal, inside = ob.plane(pos, var, rotation).intersect(origin, dir)
    elif id == 3:
        hitdist, normal, inside = ob.box(pos, var, rotation).intersect(origin, dir)
    return hitdist, normal, inside

@ti.func
def Analytical(origin, dir):
    """Analytical Method For 3D Objects \\
    Faster For Less Number Of Objects, Harder To Create Analytic Solution For 3D Objects \\
    Solution Doesn't Exist For Most 3D Objects"""
    hitdist = float('inf')
    normal = tm.vec3(0, 0, 0)
    inside = -1.0
    objects = ob.objects([0, 0, 0], [0, 0, 0], [0, 0, 0])
    material = ob.material([0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0, 0, 0, 0, 0)
    index = 0
    for i in ti.static(range(len(objectslist))):
        intersection = ObjectIntersect(objectslist[i][1], objectslist[i][2], objectslist[i][3], objectslist[i][0], origin, dir)
        if intersection[0] < (hitdist + 1e-4):
            hitdist, normal, inside = intersection
            objects = objectsfield[i]
            material = materialsfield[i]
            index = i
    return (hitdist, normal, inside, objects, material, index)
'''
@ti.func
def SDF(p, t):
    """Combine All The SDFs And Their Material Into One"""
    sdf = float('inf')
    objectdata = world.Raymarchers(p, t)
    objects = ob.plane([0, 0, 0], [0, 0, 0], [0, 0, 0])
    material = ob.material([0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0, 0, 0, 0, 0)
    for i in ti.static(range(len(objectdata))):
        objectsdf = objectdata[i]
        if objectsdf[0] < (sdf + 1e-4):
            sdf, boundingobject, material = objectsdf
            objects = boundingobject
    return (sdf, objects, material)

# https://iquilezles.org/articles/normalsSDF/
@ti.func
def CalculateNormals(pos, t):
    """Tetrahedron Technique Of Normal Approximation"""
    h = 1e-4
    kxyy = tm.vec3(1.0, -1.0, -1.0)
    kyyx = tm.vec3(-1.0, -1.0, 1.0)
    kyxy = tm.vec3(-1.0, 1.0, -1.0)
    kxxx = tm.vec3(1.0, 1.0, 1.0)
    return tm.normalize(kxyy * SDF(pos + kxyy * h, t)[0] +
                        kyyx * SDF(pos + kyyx * h, t)[0] +
                        kyxy * SDF(pos + kyxy * h, t)[0] +
                        kxxx * SDF(pos + kxxx * h, t)[0])

@ti.func
def RayMarch(origin, dir, t):
    """RayMarch Using Sphere Tracing \\
    Very Slow But Can Create Cool And Complicated 3D Shapes"""
    MinStep = 1e-4
    MinDist = 1e-4
    maxdist = 1000.0
    Iterations = 300
    hitdist = float('inf')
    t = 0.0
    delta_t = SDF(origin + dir * t, t)[0]
    for i in range(Iterations):
        t += tm.max(tm.max(delta_t, -delta_t), MinStep)
        delta_t = SDF(origin + dir * t, t)[0]
        if (t > maxdist) or (tm.max(delta_t, -delta_t) < MinDist):
            break
    if t < maxdist:
        hitdist = t
    normal = CalculateNormals(origin + dir * hitdist, t)
    inside = -1.0
    if SDF(origin, t)[0] < 0.0:
        inside = 1.0
    objects = SDF(origin + dir * hitdist, t)[1]
    material = SDF(origin + dir * hitdist, t)[2]
    if inside == 1.0:
        normal = -normal
    return (hitdist, normal, inside, objects, material)

@ti.func
def Hybrid(origin, dir, t):
    """Combine Analytical Method And RayMarching Method"""
    analytical = Analytical(origin, dir)
    raymarch = RayMarch(origin, dir, t)
    hitdist, normal, inside, objects, material = analytical[0:5]
    if raymarch[0] < (analytical[0] - 1e-4):
        hitdist, normal, inside, objects, material = raymarch
    return (hitdist, normal, inside, objects, material)
'''
# https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors#CoordinateSystem
@ti.func
def BuildCoordinateSystemFromVector(v1):
    """Build The Coordinate System On The Basis Of The Vector"""
    v2 = tm.vec3(0.0, v1[2], -v1[1]) / tm.sqrt(v1[1] * v1[1] + v1[2] * v1[2])
    if tm.max(v1[0], -v1[0]) > tm.max(v1[1], -v1[1]):
        v2 = tm.vec3(-v1[2], 0.0, v1[0]) / tm.sqrt(v1[0] * v1[0] + v1[2] * v1[2])
    v3 = tm.cross(v1, v2)
    return (v2, v3)

# https://www.pbr-book.org/3ed-2018/Color_and_Radiometry/Working_with_Radiometric_Integrals#SphericalDirection
@ti.func
def SphericalCoordsToDirection(sin_t, cos_t, phi, i, j, k):
    """Convert Spherical Coordinates To Direction"""
    return (sin_t * tm.cos(phi) * i) + (sin_t * tm.sin(phi) * j) + (cos_t * k)

# http://orbit.dtu.dk:80/fedora/objects/orbit:113874/datastreams/file_75b66578-222e-4c7d-abdf-f7e255100209/content
@ti.func
def OrthonormalBasis(n):
    """Creates Orthogonal Vectors To Each Other And Normal"""
    b1 = tm.vec3(0.0, -1.0, 0.0)
    b2 = tm.vec3(-1.0, 0.0, 0.0)
    if n[2] >= -0.9999999:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1 = tm.vec3(1.0 - (n[0] * n[0] * a), b, -n[0])
        b2 = tm.vec3(b, 1.0 - (n[1] * n[1] * a), -n[1])
    return b1, b2

@ti.func
def ToLocal(v, n):
    """Convert World Space To Local Space Based On Normal"""
    s, t = BuildCoordinateSystemFromVector(n)
    return tm.vec3(tm.dot(v, s), tm.dot(v, t), tm.dot(v, n))

@ti.func
def ToWorld(v, n):
    """Convert Local Space To World Space Based On Normal"""
    s, t = BuildCoordinateSystemFromVector(n)
    return s * v[0] + t * v[1] + n * v[2]

'''
# Generate PCG Random Numbers
# Doesn't Work Well For Vulkan And OpenGL Device When Random Numbers Are Used For Cosine Weighted Distribution Inside TraceRay Function

@ti.dataclass
class PCG32:
    state: ti.uint32

    # https://www.pcg-random.org/
    @ti.func
    def random_uint(self) -> ti.uint32:
        self.state = self.state * ti.uint32(747796405) + ti.uint32(2891336453)
        word = ((self.state >> ((self.state >> ti.uint32(28)) + ti.uint32(4))) ^ self.state) * ti.uint32(277803737)
        return (word >> ti.uint32(22)) ^ word
    
    @ti.func
    def random_float(self) -> ti.float32:
        return self.random_uint() / float(0xFFFFFFFF)

@ti.dataclass
class PCG64:
    state: ti.uint64

    # https://www.pcg-random.org/
    @ti.func
    def random_uint(self) -> ti.uint64:
        self.state = self.state * ti.uint64(6364136223846793005) + ti.uint64(1442695040888963407)
        word = ((self.state >> ((self.state >> ti.uint64(59)) + ti.uint64(5))) ^ self.state) * ti.uint64(12605985483714917081)
        return (word >> ti.uint64(43)) ^ word

    @ti.func
    def random_float(self) -> ti.float64:
        return self.random_uint() / float(0xFFFFFFFFFFFFFFFF)

@ti.func
def RandomNormal(initrand):
    return tm.cos(2.0 * tm.pi * initrand.random_float()) * tm.sqrt(-2.0 * tm.log(initrand.random_float()) / tm.log(10.0))
'''

@ti.func
def RandomPointsInDisk():
    """Generate Random Points Within Unit Disk"""
    a = 2 * tm.pi * ti.random()
    b = tm.sqrt(ti.random())
    x = b * tm.cos(a) / 10.0
    y = b * tm.sin(a) / 10.0
    return tm.vec3(x, y, 0.0)

@ti.func
def UniformRandomPointsUnitSphere():
    """Generate Random Points On Unit Sphere"""
    x = ti.randn()
    y = ti.randn()
    z = ti.randn()
    return tm.normalize(tm.vec3(x, y, z))

@ti.func
def RandomUniformDirectionHemisphere(normal):
    """Generate Uniformly Distributed Random Vectors Within Normals Hemisphere"""
    randomdir = UniformRandomPointsUnitSphere()
    randomdir *= tm.sign(tm.dot(randomdir, normal))
    return randomdir

@ti.func
def RandomCosineDirectionHemisphere(normal):
    """Generate Cosine Distributed Random Vectors Within Normals Hemisphere"""
    sumvector = normal + UniformRandomPointsUnitSphere()
    return tm.normalize(sumvector)

@ti.func
def UniformDirectionPDF():
    """Idea: Integrating Over Uniform PDF Must Give 1
    Solution: Divide 1.0 With 2 * Pi
    Note: Integrating Equation Is Solid Angle. Must Insert PDF Inside Integrals To Normalize"""
    return 1.0 / (2.0 * tm.pi)

@ti.func
def CosineDirectionPDF(costheta):
    """Idea: Integrating Over Cosine PDF Must Give 1
    Solution: Divide Cosine With Pi
    Note: Integrating Equation Is Solid Angle. Must Insert PDF Inside Integrals To Normalize"""
    return costheta / tm.pi

# https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
@ti.func
def SampleSolidAngleSphere(hitpos, centerpos, radius):
    """Samples Points On Sphere Subtended In The Solid Angle"""
    # Compute Coordinate System For Sphere Sampling
    u = tm.vec2(ti.random(), ti.random())
    refP = hitpos - centerpos
    pCenter = tm.vec3(0.0, 0.0, 0.0)
    dirc = tm.normalize(pCenter - refP)
    dircX, dircY = BuildCoordinateSystemFromVector(dirc)
    # Check Whether Point Is Inside Or Outside The Sphere
    distance2 = tm.dot(refP - pCenter, refP - pCenter)
    normalObj = UniformRandomPointsUnitSphere()
    if distance2 > radius * radius:
        # Compute Theta And Phi For Sample In Cone
        sinthetamax2 = radius * radius / distance2
        costhetamax = tm.sqrt(tm.max(1.0 - sinthetamax2, 0.0))
        costheta = (1 - u[0]) + u[0] * costhetamax
        sintheta = tm.sqrt(tm.max(1.0 - costheta * costheta, 0.0))
        phi = 2.0 * tm.pi * u[1]
        # Compute Angle Alpha From Center Of Sphere To Sample Point On Surface
        dc = tm.length(refP - pCenter)
        ds = dc * costheta - tm.sqrt(tm.max(radius * radius - dc * dc * sintheta * sintheta, 0.0))
        cosalpha = (dc * dc + radius * radius - ds * ds) / (2.0 * dc * radius)
        sinalpha = tm.sqrt(tm.max(1.0 - cosalpha * cosalpha, 0.0))
        # Compute Surface Normal And Sampled Point On Sphere
        normalObj = SphericalCoordsToDirection(sinalpha, cosalpha, phi, -dircX, -dircY, -dirc)
    positionObj = radius * normalObj
    return (tm.normalize(normalObj), positionObj + centerpos)

# https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
@ti.func
def SpherePDF(hitpos, centerpos, radius, normL, dirL):
    """Solid Angle Sphere PDF"""
    refP = hitpos - centerpos
    pCenter = tm.vec3(0.0, 0.0, 0.0)
    radius2 = radius * radius
    distance2 = tm.dot(refP - pCenter, refP - pCenter)
    costhetaL = tm.dot(normL, -dirL)
    pdf = 4.0 * tm.pi * radius2 * tm.max(costhetaL, -costhetaL) / distance2
    # Cone PDF If Point Is Outside The Sphere
    if distance2 > radius2:
        sinthetamax2 = radius2 / distance2
        costhetamax = tm.sqrt(tm.max(1.0 - sinthetamax2, 0.0))
        uniformconepdf = 1.0 / (2.0 * tm.pi * (1.0 - costhetamax))
        pdf = uniformconepdf
    return pdf

@ti.func
def PickOneLightSource():
    """Picks A Light Source Randomely And Outputs Object Index Of The Light Source"""
    lightindex = lightsfield[int(ti.floor(ti.random() * lightsfield.shape[0]))]
    return lightindex

@ti.func
def PickOneLightSourcePDF():
    """PDF Of Picking A Light Source"""
    return 1.0 / lightsfield.shape[0]

@ti.func
def Refract(I, N, eta):
    """Calculates Refraction Vector"""
    k = 1 - eta * eta * (1 - tm.dot(N, I) * tm.dot(N, I))
    R = I * 0
    if k < 0:
        # Total Internal Reflection
        R = tm.reflect(I, N)
    else:
        R = eta * I - (eta * tm.dot(N, I) + tm.sqrt(k)) * N
    return R

@ti.func
def FresnelEffect(n1, n2, w):
    """Frensel Effect Using Frensel Equations"""
    angle = tm.acos(-(w[2]))
    cosx = tm.cos(angle)
    cosy = (n1 / n2) * tm.sin(angle)
    r = 0.0
    if cosy > 1.0:
        r = 1.0
    cosy = tm.sqrt(1.0 - (cosy * cosy))
    Rs = (((n1 * cosx) - (n2 * cosy)) / ((n1 * cosx) + (n2 * cosy))) ** 2
    Rp = (((n1 * cosy) - (n2 * cosx)) / ((n1 * cosy) + (n2 * cosx))) ** 2
    r = (Rs + Rp) / 2
    return r

@ti.func
def SpectralPowerDistribution(l, l_peak, d, invert):
    """Spectral Power Distribution Function Calculated On The Basis Of Peak Wavelength And Standard Deviation \\
    Using Gaussian Function To Predict Spectral Radiance \\
    In Reality, Spectral Radiance Function Has Different Shapes For Different Objects Also Looks Much Different Than This"""
    radiance = tm.exp(-((l - l_peak) / (2 * d * d)) ** 2)
    radiance = tm.mix(radiance, 1.0 - radiance, invert)
    return radiance

@ti.func
def RefractiveIndexWavelength(l, n, l_n, s):
    """My Own Function For Refractive Index \\
    Function Is Based On Observation How Graph Of Mathematrical Functions Look Like \\
    Made To Produce Change In Refractive Index Based On Wavelength"""
    return s * ((l_n / l) - 1.0) + n

'''
@ti.func
def TraceRay(origin, dir, l, rayradiance, radiance, t, length):
    # Trace Ray
    origin_0 = tm.vec3(0, 0, 0)
    dir_0 = tm.vec3(0, 0, 0)
    hitdist, normal, inside, objects, material = Analytical(origin, dir, t)[0:5]
    isHit = hitdist < float('inf')
    isTerminate = False
    rand = ti.random()
    scatterprobability = 0.0
    index = RefractiveIndexWavelength(l, material.refractiveindex[0], material.refractiveindex[1], material.refractiveindex[2])
    if inside == 1.0:
        index = 1.0 / index
        # Absorption
        absorption_coefficient = SpectralRadiance(l, material.absorption[0], material.absorption[1], material.absorption[2]) * material.absorption[3]
        rayradiance *= tm.exp(-hitdist * absorption_coefficient)
        # Scattering
        scattering_coefficient = SpectralRadiance(l, material.scattering[0], material.scattering[1], material.scattering[2]) * material.scattering[3]
        scatterprobability = tm.exp(-hitdist * scattering_coefficient)
    # Completely Wrong Scattering :(
    if (inside == 1.0) and (rand > scatterprobability):
        t = ti.random()
        dir_0 = UniformRandomPointsUnitSphere()
        hitpos = origin + dir_0 * t
        origin_0 = hitpos
    else:
        if isHit:
            hitpos = origin + dir * hitdist
            isRough = rand < material.roughness
            isCoatRough = rand < material.coatroughness
            isCoated = rand < material.coatintensity
            isTransmit = (rand < material.transmittance) and (isCoated == False)
            fresnelchance = FresnelEffect(1.0, index, dir, normal)
            isFresnel = rand < fresnelchance
            lightemission = SpectralRadiance(l, material.emission[0], material.emission[1], material.emission[2]) * material.luminosity
            if material.luminosity > 0.0:
                radiance = rayradiance * lightemission
            dir_diffuse = RandomCosineDirectionHemisphere(normal)
            dir_specular = tm.reflect(dir, normal)
            materialreflectdir = tm.mix(dir_specular, dir_diffuse, material.roughness)
            coatreflectdir = tm.mix(dir_specular, dir_diffuse, material.coatroughness)
            refractdir = tm.mix(Refract(dir, normal, eta=(1.0 / index)), -dir_diffuse, material.roughness)
            origin_0 = tm.mix(hitpos + normal * 1e-4, hitpos - normal * 1e-4, isTransmit)
            origin_0 = tm.mix(origin_0, hitpos + normal * 1e-4, isFresnel)
            dir_0 = tm.mix(materialreflectdir, coatreflectdir, isCoated)
            dir_0 = tm.mix(dir_0, refractdir, isTransmit)
            dir_0 = tm.mix(dir_0, dir_specular, isFresnel)
            reflectradiance = tm.mix(SpectralRadiance(l, material.reflection[0], material.reflection[1], material.reflection[2]), SpectralRadiance(l, material.coat[0], material.coat[1], material.coat[2]), isCoated)
            if (isCoated and isCoatRough) or ((isCoated == False) and isRough):
                lambertianradiance = reflectradiance
                cos_t = tm.dot(dir_0, normal)
                lambertianradiance *= cos_t
                # Lambertian BRDF
                lambertianradiance /= tm.pi
                # PDF
                lambertianradiance /= CosineDirectionPDF(cos_t)
                reflectradiance = lambertianradiance
            rayradiance *= tm.mix(reflectradiance, 1.0, isTransmit)
            rayradiance = tm.mix(rayradiance, 1.0, isFresnel)
            # Russian Roulette
            rayprobability = rayradiance
            if rand >= rayprobability:
                isTerminate = True
            rayradiance *= 1.0 / rayprobability
        else:
            radiance = world.Sky(dir, l) * rayradiance
    return (radiance, rayradiance, origin_0, dir_0, isHit, isTerminate)
'''

@ti.func
def Emit(l, material):
    """Calculates Light Emittance Based On Given Material"""
    lightemission = SpectralPowerDistribution(l, material.emission[0], material.emission[1], material.emission[2]) * material.luminosity
    return lightemission

# https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models#eq:tr-d-function
@ti.func
def MicrofacetGGXDistribution(wh, alphax, alphay):
    """GGX Distribution For Microfacets"""
    cos2theta = wh[2] * wh[2]
    cos4theta = cos2theta * cos2theta
    sin2theta = tm.max(1.0 - cos2theta, 0.0)
    tan2theta = sin2theta / cos2theta
    sintheta = tm.sqrt(sin2theta)
    cosphi = ti.select(sintheta == 0.0, 1.0, tm.clamp(wh[0] / sintheta, -1.0, 1.0))
    sinphi = ti.select(sintheta == 0.0, 0.0, tm.clamp(wh[1] / sintheta, -1.0, 1.0))
    cos2phi = cosphi * cosphi
    sin2phi = sinphi * sinphi
    e = tan2theta * ((cos2phi / (alphax * alphax)) + (sin2phi / (alphay * alphay)))
    d = 1.0 / (tm.pi * alphax * alphay * cos4theta * tm.pow(1.0 + e, 2.0))
    return d

@ti.func
def MicrofacetGGXLambda(w, alphax, alphay):
    """GGX Lambda Calculation For Microfacet Geometry"""
    cos2theta = w[2] * w[2]
    sin2theta = tm.max(1.0 - cos2theta, 0.0)
    tan2theta = sin2theta / cos2theta
    sintheta = tm.sqrt(sin2theta)
    cosphi = ti.select(sintheta == 0.0, 1.0, tm.clamp(w[0] / sintheta, -1.0, 1.0))
    sinphi = ti.select(sintheta == 0.0, 0.0, tm.clamp(w[1] / sintheta, -1.0, 1.0))
    cos2phi = cosphi * cosphi
    sin2phi = sinphi * sinphi
    alpha2 = cos2phi * alphax * alphax + sin2phi * alphay * alphay
    l = (-1.0 + tm.sqrt(1.0 + (alpha2 * tan2theta))) / 2.0
    return l

@ti.func
def MicrofacetGGXGeometry(wi, wo, alphax, alphay):
    """Calculation Of GGX Masking And Shadowing Of Microfacet Geometry"""
    return 1.0 / (1.0 + MicrofacetGGXLambda(wo, alphax, alphay) + MicrofacetGGXLambda(wi, alphax, alphay))

@ti.func
def MicrofacetSpecularBRDF(wi, wo, index, roughness):
    """Microfacet Specular BRDF"""
    # Half Vector
    wh = tm.normalize(wi + wo)
    costhetaI = ti.abs(wi[2])
    costhetaO = ti.abs(wo[2])
    specular = 0.0
    # Avoid NaN
    if ((((costhetaI == 0.0) or (costhetaO == 0.0)) or ((wh[0] == 0.0) and (wh[1] == 0.0) and (wh[2] == 0.0))) == False):
        specular = (MicrofacetGGXDistribution(wh, roughness, roughness) * FresnelEffect(1.0, index, wi) * MicrofacetGGXGeometry(wi, wo, roughness, roughness)) / (4.0 * costhetaO * costhetaI)
    return specular

@ti.func
def EvaluateBRDF(l, material, dir_i, dir_o, normal, index):
    """Evaluate The BRDF"""
    # Lambertian BRDF For Diffuse Surface
    diffuse = SpectralPowerDistribution(l, material.reflection[0], material.reflection[1], material.reflection[2]) / tm.pi
    # Convert World Space To Local Space
    localindir = ToLocal(dir_i, normal)
    localoutdir = ToLocal(dir_o, normal)
    roughness = material.roughness
    specular = MicrofacetSpecularBRDF(localindir, -localoutdir, index, roughness)
    return tm.mix(diffuse, specular, material.reflectance)

@ti.func
def SampleBRDF(dir_i, normal):
    """Sample Directions Of BRDF"""
    dir_o = RandomCosineDirectionHemisphere(normal)
    return dir_o

@ti.func
def BRDFPDF(dir_o, normal):
    """PDF For Sampling Directions OF BRDF"""
    return CosineDirectionPDF(tm.dot(dir_o, normal))

@ti.func
def PowerHeuristic(pdf1, pdf2, beta):
    """Power Heuristic For Multiple Importance Sampling"""
    return tm.pow(pdf1, beta) / (tm.pow(pdf1, beta) + tm.pow(pdf2, beta))

@ti.func
def SampleLightSource(l, material, hitpos, normal, dir_i, index):
    """Sample The Light Source From The Surface"""
    # Pick A Light Source
    lightindex = PickOneLightSource()
    lightpickpdf = PickOneLightSourcePDF()
    # Get Data Of The LIght Source
    light = objectsfield[lightindex]
    # Sample Points On Light Source
    normL, posL = SampleSolidAngleSphere(hitpos + normal * 1e-4, light.pos, light.var[0])
    # Calculate The Direction Of The Ray
    dirL = tm.normalize(posL - hitpos)
    intersection = Analytical(hitpos, dirL)
    materialL, indexL = intersection[4:6]
    # Check Whether The Light Source Is Occluded By Any Object
    isVisible = indexL == lightindex
    # Calculate Light And BRDF PDFs
    lightpdf = SpherePDF(hitpos, light.pos, light.var[0], normL, dirL) * lightpickpdf
    BRDFpdf = BRDFPDF(dirL, normal)
    # Calculate The MIS Weight For Light Sampling
    MISLightWeight = PowerHeuristic(lightpdf, BRDFpdf, 2.0)
    #MISLightWeight = 0.0
    # Evaluate The BRDF
    costhetaL = tm.max(tm.dot(dirL, normal), 0.0)
    rayradiance = EvaluateBRDF(l, material, dir_i, dirL, normal, index) * costhetaL / lightpdf
    # Light Sampling
    radiance = Emit(l, materialL) * rayradiance * MISLightWeight
    return radiance * isVisible

@ti.func
def TraceRay(origin_i, dir_i, l, rayradiance, length):
    """Traces A Ray"""
    # Trace Ray
    radiance = 0.0
    hitdist, normal, inside, objects, material = Analytical(origin_i, dir_i)[0:5]
    origin_o = origin_i
    dir_o = dir_i
    isHit = hitdist < float('inf')
    isTerminate = False
    rand = ti.random()
    index = RefractiveIndexWavelength(l, material.refractiveindex[0], material.refractiveindex[1], material.refractiveindex[2])
    if inside == 1.0:
        index = 1.0 / index
    if isHit:
        origin_o = origin_i + dir_i * hitdist
        dir_o = SampleBRDF(dir_i, normal)
        BRDFpdf = BRDFPDF(dir_o, normal)
        # If The Ray Hits The Light Source
        if material.luminosity > 0.0:
            # Terminate The Path If The Ray Hits The Light Source
            isTerminate = True
            # Camera To Light
            if length < 2:
                radiance += Emit(l, material) * rayradiance
            # Calculate The Amount Of Light From The Light Source Recieved By The Object
            if length > 1:
                # BRDF Sampling
                if MIS:
                    # Calculate The MIS Weight For BRDF Sampling
                    lightpdf = 1.0 / PickOneLightSourcePDF()
                    MISBRDFWeight = PowerHeuristic(BRDFpdf, lightpdf, 2.0)
                    radiance += Emit(l, material) * rayradiance * MISBRDFWeight
                else:
                    radiance += Emit(l, material) * rayradiance
        elif MIS:
            # Sample On Objects Which Are NOT Light Source
            # Explicit Connection To The Light Source
            lightradiance = SampleLightSource(l, material, origin_o, normal, dir_i, index)
            radiance += lightradiance * rayradiance
        # Evaluate the BRDF
        costheta = tm.dot(dir_o, normal)
        rayradiance *= EvaluateBRDF(l, material, dir_i, dir_o, normal, index) * costheta / BRDFpdf
        # Russian Roulette
        if length > 3:
            rayprobability = tm.clamp(rayradiance, 0.0, 0.99)
            if rand >= rayprobability:
                isTerminate = True
            rayradiance *= 1.0 / rayprobability
    else:
        radiance = world.Sky(dir_i, l) * rayradiance
        isTerminate = True
    return (radiance, rayradiance, origin_o, dir_o, isHit, isTerminate)

@ti.func
def BuildPath(origin, dir, l):
    """Build A Unidirectional Path Using Rays"""
    # Combine Rays, Build A Path
    radiance = 0.0
    rayradiance_0 = 1.0
    origin_0 = origin
    dir_0 = dir
    isHit = False
    for length in range(path_length):
        ray_temp = TraceRay(origin_0, dir_0, l, rayradiance_0, length+1)
        radiance += ray_temp[0]
        rayradiance_0, origin_0, dir_0, isHit, isTerminate = ray_temp[1:6]
        # Terminate Rays Which Goes To Void
        if not isHit:
            break
        if isTerminate:
            break
    return radiance

@ti.func
def Scene(x, y, origin, theta, dof, lens_roughness, lens_index):
    """Color Calculation Of The Pixel Of The Whole Scene"""
    u = (2 * (x / image_width) - 1) * aspect_ratio * tan_fov
    v = (2 * (y / image_height) - 1) * tan_fov
    # SSAA
    anti_alias = ((ti.random() - 0.5) / image_width, (ti.random() - 0.5) / image_height)
    u += anti_alias[0]
    v += anti_alias[1]
    # http://www.songho.ca/opengl/gl_anglestoaxes.html
    forward = tm.vec3(tm.cos(theta[0]) * tm.cos(theta[1]), tm.sin(theta[1]), -tm.sin(theta[0]) * tm.cos(theta[1]))
    up = tm.vec3(-tm.cos(theta[0]) * tm.sin(theta[1]), tm.cos(theta[1]), tm.sin(theta[0]) * tm.sin(theta[1]))
    left = tm.cross(up, forward)
    m1 = tm.mat3([left[0], up[0], forward[0]], [left[1], up[1], forward[1]], [left[2], up[2], forward[2]])
    m2 = tm.vec3(u, v, 1)
    ray_direction = tm.normalize(tm.vec3(m1 @ m2))
    # Lens Roughness
    ray_direction = tm.mix(ray_direction, RandomCosineDirectionHemisphere(ray_direction), lens_roughness)
    color = tm.vec3(0, 0, 0)
    l = SampleSpectral(380.0, 720.0, ti.random())
    l_new = l
    step_paths_num = step_paths
    if Renderimage == True:
        if (num_paths - step_paths_num) < step_paths_num:
            step_paths_num = num_paths - step_paths_num
    ti.loop_config(parallelize=10)
    for k in range(step_paths_num):
        if k > 0:
            # Hero Wavelength Spectral Sampling WIthout MIS :(
            l_new = HeroSpectralSampling(l, k, 380.0, 720.0, step_paths_num - 1)
        # Lens Refraction, Dispersion(Chromatic Abberation)
        ori = origin
        dir = ray_direction
        dir = Refract(dir, -forward, eta=(1 / RefractiveIndexWavelength(l_new, lens_index[0], lens_index[1], lens_index[2])))
        if depth_of_field:
            # Depth Of Field
            random_points_disk = RandomPointsInDisk() * dof[0]
            ori += random_points_disk[0] * left + random_points_disk[1] * up
            lookat = origin + dof[1] * dir
            dir = tm.normalize(lookat - ori)
        color += (BuildPath(ori, dir, l_new) * spectrum.WaveToXYZ(l_new)) / SpectralPDF(380.0, 720.0)
    color /= step_paths_num
    return color

@ti.func
def CalculateOrigin(origin, delta_origin, theta):
    """Calculates New Origin Based On Given Parameters"""
    # http://www.songho.ca/opengl/gl_anglestoaxes.html
    forward = tm.vec3(tm.cos(theta[0]) * tm.cos(theta[1]), tm.sin(theta[1]), -tm.sin(theta[0]) * tm.cos(theta[1]))
    up = tm.vec3(-tm.cos(theta[0]) * tm.sin(theta[1]), tm.cos(theta[1]), tm.sin(theta[0]) * tm.sin(theta[1]))
    left = tm.cross(up, forward)
    origin += delta_origin[0] * forward + delta_origin[1] * up + delta_origin[2] * left
    return origin


# Kernel
@ti.kernel
def Kernel(s:ti.uint32, origin:tm.vec3, delta_origin:tm.vec3, theta:tm.vec2, t:ti.f32, world:ti.template()) -> tm.vec3:
    dof, lens_roughness, lens_index = world.Camera(t)[2:5]
    new_origin = CalculateOrigin(origin, delta_origin, theta)
    for x, y in pixels:
        pixels[x, y] += tm.max(Scene(x, y, new_origin, theta, dof, lens_roughness, lens_index), 0.0)
    for x, y in imagepixels:
        imagepixels[x, y] = Gamma(BioPhotometricTonemap(tm.max(spectrum.XYZToRGB(pixels[x, y] / s), 0.0)), 2.2)
    return new_origin


# Python Environment
def UpdateWorldStuff(world, t, ray_origin, origin, theta, w_theta):
    """Update Camera Data Based On World Data"""
    cameradata = world.Camera(t)
    origin -= ray_origin
    ray_origin = tm.vec3(cameradata[0])
    theta -= w_theta
    w_theta = tm.vec2(cameradata[1]) * tm.pi / 180.0
    origin += ray_origin
    theta += w_theta
    return (ray_origin, origin, theta, w_theta)

def STRToID(string):
    id = 0
    if string == 'sphere':
        id = 1
    elif string == 'plane':
        id = 2
    elif string == 'box':
        id = 3
    return id

def CreateObjectsList(world, t):
    """Create A List Of Objects From The World File"""
    objects = world.Intersectors(t)
    objectslist = []
    for i in range(len((objects))):
        obj = [STRToID(objects[i][0]), objects[i][1], objects[i][2], objects[i][3], objects[i][4]]
        objectslist.append(obj)
    return objectslist

def CreateLightsList(objectslist):
    """Create A List Of All Light Sources"""
    lightslist = []
    for num in range(len(objectslist)):
        if objectslist[num][4].luminosity > 0.0:
            lightslist.append(num)
    return lightslist

def InitializeObjectFields(objectslist, lightslist):
    """Initialize Object Fields"""
    objectsfield = ob.objects.field(shape=(len(objectslist),))
    materialsfield = ob.material.field(shape=(len(objectslist),))
    lightsfield = ti.field(dtype=ti.i32, shape=(len(lightslist),))
    for num in range(len(lightslist)):
        lightsfield[num] = lightslist[num]
    for obj in range(len(objectslist)):
        objectsfield[obj].pos = objectslist[obj][1]
        objectsfield[obj].var = objectslist[obj][2]
        objectsfield[obj].rotation = objectslist[obj][3]
        objectsfield[obj].id = objectslist[obj][0]
        materialsfield[obj].reflection = objectslist[obj][4].reflection
        materialsfield[obj].absorption = objectslist[obj][4].absorption
        materialsfield[obj].scattering = objectslist[obj][4].scattering
        materialsfield[obj].coat = objectslist[obj][4].coat
        materialsfield[obj].emission = objectslist[obj][4].emission
        materialsfield[obj].refractiveindex = objectslist[obj][4].refractiveindex
        materialsfield[obj].roughness = objectslist[obj][4].roughness
        materialsfield[obj].reflectance = objectslist[obj][4].reflectance
        materialsfield[obj].coatroughness = objectslist[obj][4].coatroughness
        materialsfield[obj].coatintensity = objectslist[obj][4].coatintensity
        materialsfield[obj].transmittance = objectslist[obj][4].transmittance
        materialsfield[obj].luminosity = objectslist[obj][4].luminosity
    return objectsfield, materialsfield, lightsfield

if Renderimage == True:
    if animation == True:
        t = (frames_start - 1) / fps_0
        start = time.perf_counter()
        worldspec = importlib.util.spec_from_file_location("world", worldfilepath.name)
        world = importlib.util.module_from_spec(worldspec)
        worldspec.loader.exec_module(world)
        for p in tqdm(range(frames_start, frames_end+1)):
            #worldspec = importlib.util.spec_from_file_location("world", worldfilepath.name)
            #world = importlib.util.module_from_spec(worldspec)
            #worldspec.loader.exec_module(world)
            cameradata = world.Camera(t)
            ray_origin = tm.vec3(cameradata[0])
            theta = tm.vec2(cameradata[1]) * tm.pi / 180.0
            objectslist = CreateObjectsList(world, t)
            lightslist = CreateLightsList(objectslist)
            objectsfield, materialsfield, lightsfield = InitializeObjectFields(objectslist, lightslist)
            for s in tqdm(range(1, math.ceil(num_paths / step_paths)+1)):
                Kernel(s, ray_origin, tm.vec3(0, 0, 0), theta, t, world)
            ti.tools.imwrite(imagepixels.to_numpy(), os.path.join(animdirectory, f'frame_{p}.png'))
            pixels.fill(0)
            t += 1 / fps_0
        end = time.perf_counter()
        time_taken = end - start
        print(f"The animation rendering has been completed in {round(time_taken, 4)} seconds with {num_paths} paths and path length {path_length} for each frame.")
    else:
        if ui == True:
            gui = ti.GUI("Path Tracer", res=(image_width, image_height), fast_gui=True)
        t = time_s
        worldspec = importlib.util.spec_from_file_location("world", worldfilepath.name)
        world = importlib.util.module_from_spec(worldspec)
        worldspec.loader.exec_module(world)
        cameradata = world.Camera(t)
        ray_origin = tm.vec3(cameradata[0])
        theta = tm.vec2(cameradata[1]) * tm.pi / 180.0
        objectslist = CreateObjectsList(world, t)
        lightslist = CreateLightsList(objectslist)
        objectsfield, materialsfield, lightsfield = InitializeObjectFields(objectslist, lightslist)
        start = time.perf_counter()
        for s in tqdm(range(1, math.ceil(num_paths / step_paths)+1)):
            Kernel(s, ray_origin, tm.vec3(0, 0, 0), theta, t, world)
            if ui == True:
                gui.set_image(imagepixels)
                gui.show()
        end = time.perf_counter()
        time_taken = end - start
        print(f"The rendering has been completed in {round(time_taken, 4)} seconds with {num_paths} paths and path length {path_length}.")
        ti.tools.imwrite(imagepixels.to_numpy(), Savedirectory)
        print("The rendered image has been saved.")
else:
    gui = ti.GUI("Path Tracing", res=(image_width, image_height), fast_gui=True)
    s = 0
    ray_origin = tm.vec3(0.0, 0.0, 0.0)
    origin = tm.vec3(0.0, 0.0, 0.0)
    d_origin = tm.vec3(0, 0, 0)
    cursor_pos_0 = tm.vec2(0, 0)
    cursor_pos_1 = tm.vec2(0, 0)
    theta = tm.vec2(0.0, 0.0)
    w_theta = tm.vec2(0.0, 0.0)
    d_theta = tm.vec2(0.0, 0.0)
    frame_start = time.perf_counter()
    frame_end = 0
    fps = 60
    t_start = time.perf_counter()
    t = 0.0
    worldspec = importlib.util.spec_from_file_location("world", worldfilepath.name)
    world = importlib.util.module_from_spec(worldspec)
    worldspec.loader.exec_module(world)
    ray_origin, origin, theta, w_theta = UpdateWorldStuff(world, t, ray_origin, origin, theta, w_theta)
    objectslist = CreateObjectsList(world, t)
    lightslist = CreateLightsList(objectslist)
    objectsfield, materialsfield, lightsfield = InitializeObjectFields(objectslist, lightslist)
    while gui.running:
        gui.get_event()
        if gui.is_pressed('k'):
            t_end = time.perf_counter()
            t += (t_end - t_start)
            t_start = time.perf_counter()
            ray_origin, origin, theta, w_theta = UpdateWorldStuff(world, t, ray_origin, origin, theta, w_theta)
            objectslist = CreateObjectsList(world, t)
        else:
            t_start = time.perf_counter()
        if gui.is_pressed('r'):
            worldspec = importlib.util.spec_from_file_location("world", worldfilepath.name)
            world = importlib.util.module_from_spec(worldspec)
            worldspec.loader.exec_module(world)
            ray_origin, origin, theta, w_theta = UpdateWorldStuff(world, t, ray_origin, origin, theta, w_theta)
            objectslist = CreateObjectsList(world, t)
            lightslist = CreateLightsList(objectslist)
            objectsfield, materialsfield, lightsfield = InitializeObjectFields(objectslist, lightslist)
            pixels.fill(0)
            s = 0
        press = gui.is_pressed(ti.GUI.LMB)
        cursor_pos_1 = gui.get_cursor_pos()
        if press:
            dx = cursor_pos_1[0] - cursor_pos_0[0]
            dy = cursor_pos_1[1] - cursor_pos_0[1]
            d_theta[0] = 3.0 * dx
            d_theta[1] = 3.0 * dy
            theta -= d_theta
            pixels.fill(0)
            s = 0
        cursor_pos_0 = cursor_pos_1
        d_origin[0] = 3.0 / fps * (gui.is_pressed('w') - gui.is_pressed('s'))
        d_origin[1] = 3.0 / fps * (gui.is_pressed('e') - gui.is_pressed('q'))
        d_origin[2] = 3.0 / fps * (gui.is_pressed('d') - gui.is_pressed('a'))
        keys = ['w', 's', 'e', 'q', 'd', 'a', 'k']
        for w in keys:
            if gui.is_pressed(w):
                pixels.fill(0)
                s = 0
        s = s + 1
        origin = Kernel(s, origin, d_origin, theta, t, world)
        gui.set_image(imagepixels)
        gui.show()
        frame_end = time.perf_counter()
        fps = 1 / (frame_end - frame_start)
        frame_start = frame_end
        print('\033[1A\033[K', end='')
        print('\033[1A\033[K', end='')
        print('\033[1A\033[K', end='')
        print('\033[1A\033[K', end='')
        print(f'Samples: {s * step_paths}')
        print(f'Camera Position: {tm.vec3(round(origin[0], 3), round(origin[1], 3), round(origin[2], 3))}')
        print(f'Camera Angle: {tm.vec2(round(theta[0] * 180.0 / tm.pi, 3), round(theta[1] * 180.0 / tm.pi, 3))}')
        print(f'Time: {round(t, 3)}')

