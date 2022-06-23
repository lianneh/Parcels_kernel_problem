import math

# define beaching status
# 0 = at sea, 1 = beached, 2 = after non-beach dyn (?), 3 = after beach dyn (???), 4 = please unbeach (not needed yet).
# if my particles get beached, I will keep them beached, for now.
# treating particle beached == 2 as "now check beaching status"

def AdvectionRK4(particle, fieldset, time):
#    """Advection of particles using fourth-order Runge-Kutta integration.
#    Taken from source code in Parcels docs but adapted following Delandmeter 2019 North Sea example.
#    Function needs to be converted to Kernel object before execution"""
    if particle.beached == 0:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.beached = 2

def DiffusionUniformKh(particle, fieldset, time):
#    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.
#
#    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
#    These can be added via e.g.
#        fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
#
#        fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)
#    where mesh is either 'flat' or 'spherical'
#
#    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
#    Since the perturbation due to diffusion is in this case isotropic independent, this
#    kernel contains no advection and can be used in combination with a seperate
#    advection kernel.
#
#    The Wiener increment `dW` is normally distributed with zero
#    mean and a standard deviation of sqrt(dt).
#    """
    # Wiener increment with zero mean and std of sqrt(dt)
    if particle.beached == 0:
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy
        particle.beached = 2

def Windage(particle,fieldset,time):
    if particle.beached == 0:
        dtt = particle.dt
        (u_wind, v_wind) = fieldset.UV_wind[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_wind * dtt
        particle.lat += v_wind * dtt
        particle.beached = 2

def StokesDrift(particle,fieldset,time):
    if particle.beached == 0:
        dtt = particle.dt
        (u_stokes, v_stokes) = fieldset.UV_stokes[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_stokes * dtt
        particle.lat += v_stokes * dtt
        particle.beached = 2

# adapted from Delandmeter 2019 North Sea Examples
# checks u and v values, if very small, particle is considered to be on land and classified as particle.beached == 1
def BeachedStatusCheck(particle, fieldset, time):
    if particle.beached == 2:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if math.fabs(u) < 1e-14 and math.fabs(v) < 1e-14: # math.fabs is absolute value as a float (abs will return an integer
            particle.beached = 1 # beached
        else:
            particle.beached = 0 # not beached

# if beached, print out location and then delete.
def BeachedDelete(particle, fieldset, time):
	if particle.beached == 1:
	    #print("Particle beached.") 
	    #print(f'particle.lon is {particle.lon}')
	    #print(f'particle.lat is {particle.lat}')
	    #print("particle.beached = " + str(particle.beached))
	    particle.delete()


