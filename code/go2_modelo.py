"""
Modelo del Unitree Go2 leído del URDF (carpeta go2_urdf).

Todos los parámetros (masas, centros de masa, inercias, orígenes de las articulaciones,
límites articulares y de torque, caja de colisión del tronco) se obtienen al parsear
go2_urdf/go2_description.urdf con xml.etree.

Convenciones:
  * Marco del cuerpo {B}: x hacia adelante, y hacia la izquierda, z hacia arriba.
  * Orden de las patas: FR, FL, RR, RL.
  * Ángulos por pata: [abducción (hip), cadera (thigh), rodilla (calf)].
    Una rotación positiva de thigh lleva el pie hacia atrás; la rodilla es negativa.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_urdf", "go2_description.urdf")
G = 9.81
PATAS = ["FR", "FL", "RR", "RL"]


# >> Lectura del URDF
def _vec(texto):
    return np.array([float(v) for v in texto.split()])


def leer_urdf(ruta=URDF):
    """Devuelve dos diccionarios: links[nombre] y joints[nombre] con los datos del URDF."""
    raiz = ET.parse(ruta).getroot()
    links, joints = {}, {}
    for l in raiz.findall("link"):
        d = {"masa": 0.0, "cdm": np.zeros(3), "I": np.zeros((3, 3)), "caja": None}
        ine = l.find("inertial")
        if ine is not None:
            d["masa"] = float(ine.find("mass").get("value"))
            o = ine.find("origin")
            d["cdm"] = _vec(o.get("xyz")) if o is not None else np.zeros(3)
            a = {k: float(v) for k, v in ine.find("inertia").attrib.items()}
            d["I"] = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                               [a["ixy"], a["iyy"], a["iyz"]],
                               [a["ixz"], a["iyz"], a["izz"]]])
        col = l.find("collision/geometry/box")
        if col is not None:
            d["caja"] = _vec(col.get("size"))
        links[l.get("name")] = d
    for j in raiz.findall("joint"):
        o = j.find("origin")
        lim = j.find("limit")
        joints[j.get("name")] = {
            "tipo": j.get("type"), "padre": j.find("parent").get("link"),
            "hijo": j.find("child").get("link"),
            "xyz": _vec(o.get("xyz")) if o is not None else np.zeros(3),
            "lim": None if lim is None else {k: float(v) for k, v in lim.attrib.items()},
        }
    return links, joints


LINKS, JOINTS = leer_urdf()

# >> Parámetros derivados
_BASE = JOINTS["FR_hip_joint"]["padre"]                   # eslabón del tronco (padre de las caderas)
M_BASE = LINKS[_BASE]["masa"]
CDM_BASE = LINKS[_BASE]["cdm"]
I_BASE = LINKS[_BASE]["I"]
M_HIP = LINKS["FR_hip"]["masa"]
M_THIGH = LINKS["FR_thigh"]["masa"]
M_CALF = LINKS["FR_calf"]["masa"]
M_FOOT = LINKS["FR_foot"]["masa"]
M_PATA = M_HIP + M_THIGH + M_CALF + M_FOOT
MASA = sum(d["masa"] for d in LINKS.values())             # masa total del robot

HIP_X, HIP_Y = JOINTS["FR_hip_joint"]["xyz"][0], abs(JOINTS["FR_hip_joint"]["xyz"][1])
L0 = abs(JOINTS["FR_thigh_joint"]["xyz"][1])              # desplazamiento lateral del muslo
L1 = abs(JOINTS["FR_calf_joint"]["xyz"][2])               # longitud del muslo
L2 = abs(JOINTS["FR_foot_joint"]["xyz"][2])               # longitud de la pierna (calf -> pie)
CUERPO = LINKS[_BASE]["caja"]                              # caja de colisión del tronco

SIGNO_X = np.array([+1, +1, -1, -1])                      # adelante / atrás
SIGNO_Y = np.array([-1, +1, -1, +1])                      # derecha / izquierda
P_ABD = np.column_stack([SIGNO_X * HIP_X, SIGNO_Y * HIP_Y, np.zeros(4)])
P_CADERA = np.column_stack([SIGNO_X * HIP_X, SIGNO_Y * (HIP_Y + L0), np.zeros(4)])

_lims = [JOINTS[f"FR_{n}_joint"]["lim"] for n in ("hip", "thigh", "calf")]
Q_MIN = np.array([l["lower"] for l in _lims])
Q_MAX = np.array([l["upper"] for l in _lims])
TAU_MAX = np.array([l["effort"] for l in _lims])
DQ_MAX = np.array([l["velocity"] for l in _lims])


def inercia_srbm(z_pies=-0.30):
    """Tensor de inercia del cuerpo rígido equivalente (SRBM) respecto del CdM.

    Tronco: inercia del URDF. Patas: cada eslabón se considera una masa puntual en la
    postura nominal (pie bajo la cadera, a una altura z_pies).
    """
    masas, pos = [M_BASE], [CDM_BASE]
    for i in range(4):
        cad = P_CADERA[i]
        masas += [M_HIP, M_THIGH, M_CALF + M_FOOT]
        pos += [P_ABD[i], cad + [0, 0, 0.25 * z_pies], cad + [0, 0, 0.75 * z_pies]]
    masas, pos = np.array(masas), np.array(pos)
    cdm = (masas[:, None] * pos).sum(0) / masas.sum()
    I = np.diag(np.diag(I_BASE))
    for mi, ri in zip(masas, pos - cdm):
        I += mi * (ri @ ri * np.eye(3) - np.outer(ri, ri))
    return I, cdm


def parametros_planos():
    """Parámetros del modelo multicuerpo PLANO (sagital) para la pregunta 2.

    Cada 'pata' plana = par izquierda + derecha (masas, inercias y torques x 2).
    Tronco plano = base + 4 motores de abducción (la abducción queda fija).
    """
    m_t = M_BASE + 4 * M_HIP
    cx_t = M_BASE * CDM_BASE[0] / m_t
    I_t = (I_BASE[1, 1] + 4 * LINKS["FR_hip"]["I"][1, 1] + 4 * M_HIP * HIP_X**2
           + M_BASE * CDM_BASE[0]**2 - m_t * cx_t**2)
    lc1 = abs(LINKS["FR_thigh"]["cdm"][2])
    lc_calf = abs(LINKS["FR_calf"]["cdm"][2])
    m2_1 = M_CALF + M_FOOT
    lc2 = (M_CALF * lc_calf + M_FOOT * L2) / m2_1
    I2_1 = (LINKS["FR_calf"]["I"][1, 1] + M_CALF * (lc_calf - lc2)**2
            + LINKS["FR_foot"]["I"][1, 1] + M_FOOT * (L2 - lc2)**2)
    return dict(m_t=m_t, cx_t=cx_t, I_t=I_t,
                m1=2 * M_THIGH, lc1=lc1, I1=2 * LINKS["FR_thigh"]["I"][1, 1],
                m2=2 * m2_1, lc2=lc2, I2=2 * I2_1,
                L1=L1, L2=L2, HIP_X=HIP_X, TRONCO=CUERPO[[0, 2]],
                TAU_MAX=2 * TAU_MAX[1:], DQ_MAX=DQ_MAX[1:], Q_MIN=Q_MIN[1:], Q_MAX=Q_MAX[1:])


# >> Cinemática de una pata
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def cd_pata(q, i):
    """Cinemática directa de la pata i en {B}: devuelve (cadera, rodilla, pie)."""
    q0, q1, q2 = q
    Rx = rot_x(q0)
    cadera = P_ABD[i] + Rx @ np.array([0, SIGNO_Y[i] * L0, 0])
    rod_plano = np.array([-L1 * np.sin(q1), 0, -L1 * np.cos(q1)])
    pie_plano = rod_plano + np.array([-L2 * np.sin(q1 + q2), 0, -L2 * np.cos(q1 + q2)])
    return cadera, cadera + Rx @ rod_plano, cadera + Rx @ pie_plano


def ci_pata(p_pie_b, i):
    """Cinemática inversa analítica de la pata i. p_pie_b: posición del pie en {B}."""
    dx, dy, dz = np.asarray(p_pie_b, float) - P_ABD[i]
    l0 = SIGNO_Y[i] * L0
    L = np.sqrt(max(dy**2 + dz**2 - l0**2, 1e-9))
    q0 = np.arctan2(dz, dy) - np.arctan2(-L, l0)
    q0 = (q0 + np.pi) % (2 * np.pi) - np.pi
    c2 = (dx**2 + L**2 - L1**2 - L2**2) / (2 * L1 * L2)
    q2 = -np.arccos(np.clip(c2, -1.0, 1.0))
    q1 = np.arctan2(-dx, L) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q0, q1, q2])


def jacobiano_pata(q, i, eps=1e-6):
    """Jacobiano 3x3 de la posición del pie en {B} (diferencias finitas centradas)."""
    J = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3); dq[k] = eps
        J[:, k] = (cd_pata(q + dq, i)[2] - cd_pata(q - dq, i)[2]) / (2 * eps)
    return J


def resumen():
    """Tabla de los parámetros leídos del URDF."""
    filas = [("Masa total [kg]", f"{MASA:.3f}"), ("Masa del tronco [kg]", f"{M_BASE:.3f}"),
             ("Masa por pata (hip/thigh/calf/foot) [kg]", f"{M_HIP}/{M_THIGH}/{M_CALF}/{M_FOOT}"),
             ("Cadera (abducción) en {B} [m]", f"(±{HIP_X}, ±{HIP_Y}, 0)"),
             ("Offset lateral del muslo L0 [m]", f"{L0}"), ("Muslo L1 / pierna L2 [m]", f"{L1} / {L2}"),
             ("Caja del tronco [m]", f"{CUERPO}"), ("q_min [rad]", f"{Q_MIN}"), ("q_max [rad]", f"{Q_MAX}"),
             ("Torque máx. [N m]", f"{TAU_MAX}"), ("Velocidad máx. [rad/s]", f"{DQ_MAX}")]
    for a, b in filas:
        print(f"{a:45s} {b}")


if __name__ == "__main__":
    resumen()
    I, c = inercia_srbm()
    print("Inercia SRBM:\n", np.round(I, 4))
    print("Parámetros planos:", {k: np.round(v, 4) for k, v in parametros_planos().items()})
    for i in range(4):
        q = np.array([0.1, 0.8, -1.6])
        assert np.allclose(ci_pata(cd_pata(q, i)[2], i), q)
    print("Cinemática inversa verificada.")
