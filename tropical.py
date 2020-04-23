import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import torch
from torch import nn
import string
from scipy.spatial import ConvexHull, Delaunay
import JuPyMake
import matplotlib.pyplot as plt

JuPyMake.InitializePolymake()
JuPyMake.ExecuteCommand("application 'tropical';")

class Tropical:

    def __init__(self, val):
        self.val = val

    # Relations
    def __lt__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val < 0
        return self.val - other < 0

    def __gt__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val > 0
        return self.val - other > 0

    def __le__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val <= 0
        return self.val - other <= 0

    def __ge__(self, other):
        if isinstance(other, Tropical):
            return self.val - other.val >= 0
        return self.val - other >= 0

    def __eq__(self, other):
        if isinstance(other, Tropical):
            return self.val == other.val
        return self.val == other

    # Simple operations
    def __add__(self, other):
        if isinstance(other, Tropical):
            return Tropical(max(self.val, other.val))
        return Tropical(max(self.val, other))

    def __radd__(self, other):
        if isinstance(other, Tropical):
            return Tropical(max(self.val, other.val))
        return Tropical(max(self.val, other))

    def __mul__(self, other):
        if isinstance(other, Tropical):
            return Tropical(self.val + other.val)
        return Tropical(self.val + other)

    def __rmul__(self, other):
        if isinstance(other, Tropical):
            return Tropical(self.val + other.val)
        return Tropical(self.val + other)

    def __pow__(self, other):
        assert float(other) == int(float(other)), 'pow should be natural'
        assert float(other) >= 0, 'pow should be natural'
        if isinstance(other, Tropical):
            return Tropical(self.val * other.val)
        return Tropical(self.val * other)

    # Other
    def __abs__(self):
        return Tropical(abs(self.val))

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    def __float__(self):
        return float(self.val)

    def sym(self):
        return Tropical(-self.val)

    def __truediv__(self, b):
        return self * b.sym()

    def __floordiv__(self, b):
        return self * b.sym()


class TropicalMonomial:

    def __init__(self, coef):
        if isinstance(coef, TropicalMonomial):
            self.coef = coef.coef
        else:
            self.coef = [Tropical(x) if not isinstance(x, Tropical) else x for x in coef]

    def __getitem__(self, n):
        return self.coef[n]

    def __len__(self):
        return len(self.coef)

    def __eq__(self, mono):
        if len(self) == len(mono):
            if all([x == y for x, y in zip(self.coef, mono.coef)]):
                return True
        return False

    def __add__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalPolynomial([self.coef, other.coef])

    def __radd__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalPolynomial([self.coef, other.coef])

    def __mul__(self, other):
        if type(other) == TropicalMonomial:
            assert len(self) == len(other)

            return TropicalMonomial([x * y for x, y in zip(self.coef, other.coef)])

        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i == 0 else x for i, x in enumerate(self.coef)])

    def __rmul__(self, other):
        if type(other) == TropicalMonomial:
            assert len(self) == len(other)

            return TropicalMonomial([x * y for x, y in zip(self.coef, other.coef)])
        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i == 0 else x for i, x in enumerate(self.coef)])

    def __div__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalMonomial([x / y for x, y in zip(self.coef, other.coef)])

    def __rdiv__(self, other):
        assert type(other) == TropicalMonomial
        assert len(self) == len(other)

        return TropicalMonomial([x / y for x, y in zip(self.coef, other.coef)])

    def __pow__(self, other):
        return TropicalMonomial([x ** other for x in self.coef])

    def __repr__(self):
        return str(self)

    def __str__(self):
        d = len(self)

        var = string.ascii_lowercase

        if d == -1:
            return '0'

        if d == 0:
            return '{}'.format(self[0])

        out = []

        for pwr in range(d):

            coe = self[pwr]

            v = var[pwr - 1]

            if coe == 0:
                continue

            if pwr == 0:
                s = '{}'.format(coe)

            else:
                if coe == 1:
                    s = '{}'.format(v)

                else:
                    s = '{}^{}'.format(v, int(coe.val))

            out.append(s)

        out = '⨀'.join(out)

        return out

    def evaluate(self, point):
        """Evaluate the monomial at a given point or points"""
        point = [Tropical(x) for x in point]
        assert len(point) == len(self) - 1

        out = self.coef[0]

        for pwr, coef in enumerate(self.coef[1:]):
            out *= point[pwr] ** coef

        return out


class TropicalPolynomial:

    def __init__(self, monoms):

        if isinstance(monoms, list):
            self.monoms = {}

            for x in monoms:
                if any(isinstance(i, Tropical) for i in x):
                    x = [i.val for i in x]

                if tuple(x[1:]) in self.monoms.keys():
                    prev_mon = self.monoms[tuple(x[1:])]
                    new_x = [prev_mon[0] + x[0]] + list(x[1:])
                    self.monoms[tuple(new_x[1:])] = TropicalMonomial(new_x)
                else:
                    self.monoms[tuple(x[1:])] = TropicalMonomial(x)
        elif isinstance(monoms, dict):
            self.monoms = monoms

    def __getitem__(self, n):
        return list(self.monoms.values())[n]

    def __len__(self):
        return len(self.monoms)

    def __eq__(self, other):
        if len(set(self.monoms.keys()).symmetric_difference(other.monoms.keys())) == 0:
            return True
        return False

    def __add__(self, other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0] + new_monom[x][0]] + list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other + self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other + self
        return TropicalPolynomial(new_monom)

    def __radd__(self, other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0] + new_monom[x][0]] + list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other + self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other + self

        return TropicalPolynomial(new_monom)

    def __mul__(self, other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    new_monom[key_z] = z
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other * self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other * self

        return TropicalPolynomial(new_monom)

    def __rmul__(self, other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    print(x,y)
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    new_monom[key_z] = z
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return other * self
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val] + [0 for _ in range(len(self[0].coef) - 1)]])
            return other * self

        return TropicalPolynomial(new_monom)

    def __pow__(self, other):
        new_monom = {}
        for y in self.monoms:
            z = self.monoms[y] ** other
            key_z = tuple([i.val for i in z.coef[1:]])
            new_monom[key_z] = z
        return TropicalPolynomial(new_monom)

    def __repr__(self):
        return str(self)

    def __str__(self):

        if len(self) == 0:
            return "0"

        out = []

        for pwr in self.monoms.keys():
            s = str(self.monoms[pwr])

            out.append(s)
            
        out = sorted(out, key=lambda x: x[::-1])

        out = ' ⨁ '.join(out)

        return out

    def evaluate(self, point):
        """Evaluate the polynomial at a given point or points"""
        out = []

        for x in self.monoms:
            out.append(self.monoms[x].evaluate(point))

        out = max(out)

        return out
    
    def minimize_depr(self, tolerance=1e-12):
        def in_hull(p, del_hull):
            return del_hull.find_simplex(p)>=0
        
        def points_in_hull(points, hull):
            eq=hull.equations.T
            V,b=eq[:-1].T,eq[-1]
            flag = np.prod(np.dot(V,points.T)+b[:,None]<= tolerance,axis=0)
            return flag.astype(bool)

        def hit(U, hull):
            U0 = np.ones(U.shape)
            U0[:,1:] *= 0
            eq=hull.equations.T
            V,b=eq[:-1].T,eq[-1]
            num = -(b[:,None] + np.dot(V,U.T))
            num[np.isclose(num,0)] *= 0
            den = np.dot(V,U0.T)
            alpha = np.divide(num,den)
            a = np.min(alpha,axis=0,initial=np.inf,where=(~np.isnan(alpha))&(~np.isinf(alpha))&(alpha>0))
            U0[:,0] = a
            pa = U + U0
            return pa
        
        def filter_hull(points,hull):
            ch = points[hull.vertices]
            hit_p = hit(ch,hull)
            in_hull = points_in_hull(hit_p, hull)
            new_ch = ch[~in_hull]
            return sorted(new_ch.tolist())
        
        pts = np.array([[i.val for i in mon.coef] for mon in self.monoms.values()])
        if len(pts)<len(pts[0]):
            return self
        hull = ConvexHull(pts,qhull_options='Qa')
        
        new_monom = filter_hull(pts,hull)
        return TropicalPolynomial(new_monom)
    
    def minimize(self):
        name = 'test'
        JuPyMake.ExecuteCommand(f'${name} = toTropicalPolynomial("{self.poly_to_str()}");')

        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>${name});')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')
        
        pts = JuPyMake.ExecuteCommand('print $ds->POINTS;')[1]
        pts = np.array([[int(j) for j in i.split()[1:]] for i in pts.split('\n')[:-1]])
        
        simp = JuPyMake.ExecuteCommand('print $ds->MAXIMAL_CELLS;')[1]
        simp = np.array([[int(j) for j in i[1:-1].split()] for i in simp.split('\n')[:-1]])
        
        adj = JuPyMake.ExecuteCommand('for (my $i=0; $i<$ds->N_MAXIMAL_CELLS; ++$i)\
                                    {print $ds->cell($i)->GRAPH->ADJACENCY, "\t" }')[1]
        new_adj = []
        for i in adj.split('\t')[:-1]:
            new_a = []
            for j in i.split('\n')[:-1]:
                kek = j[1:-1].split()
                if len(kek)>0:
                    new_a.append([int(kek[0]),int(kek[-1])])
                else:
                    new_a.append([])
            new_adj.append(new_a)

        new_simp = []
        for i, vv in enumerate(new_adj):
            new_s = []

            for j, v in enumerate(vv):
                if len(v)>0:
                    new_s.append([simp[i][j],simp[i][v[0]]])
                    new_s.append([simp[i][j],simp[i][v[1]]])
#             new_s = merge_intervals(new_s)
            new_simp.append(new_s)
            
#         print(simp)
#         print(new_simp)
        used_points = np.unique([i for j in new_simp for i in j])
        
        return TropicalPolynomial([self.monoms[tuple(v)] for v in pts[used_points]])
    

    def poly_to_str(self):
        def mon_to_str(monom):
            return str(monom.coef[0].val) + '+' +''.join([str(int(v.val))+'*'+'x{}+'.format(i) for i,v in enumerate(monom.coef[1:])])[:-1]
        
        s = 'max('+','.join([mon_to_str(mon) for mon in self.monoms.values()]) + ')'
        return s

    def plot_dual_sub(self, color='blue', name='a'):
        JuPyMake.ExecuteCommand(f'${name} = toTropicalPolynomial("{self.poly_to_str()}");')

        JuPyMake.ExecuteCommand(f'$V = new Hypersurface<Max>(POLYNOMIAL=>${name});')
        JuPyMake.ExecuteCommand('$ds = $V->dual_subdivision();')

        pts = JuPyMake.ExecuteCommand('print $ds->POINTS;')[1]
        pts = np.array([[int(j) for j in i.split()[1:]] for i in pts.split('\n')[:-1]])

        plt.plot(pts[:,0], pts[:,1], 'o', color='black')

        simp = JuPyMake.ExecuteCommand('print $ds->MAXIMAL_CELLS;')[1]
        simp = np.array([[int(j) for j in i[1:-1].split()] for i in simp.split('\n')[:-1]])

        adj = JuPyMake.ExecuteCommand('for (my $i=0; $i<$ds->N_MAXIMAL_CELLS; ++$i)\
                                    {print $ds->cell($i)->GRAPH->ADJACENCY, "\t" }')[1]
        new_adj = []
        for i in adj.split('\t')[:-1]:
            new_a = []
            for j in i.split('\n')[:-1]:
                kek = j[1:-1].split()
                if len(kek)>0:
                    new_a.append([int(kek[0]),int(kek[-1])])
                else:
                    new_a.append([])
            new_adj.append(new_a)

        new_simp = []
        for i, vv in enumerate(new_adj):
            new_s = []

            for j, v in enumerate(vv):
                if len(v)>0:
                    new_s.append([simp[i][j],simp[i][v[0]]])
                    new_s.append([simp[i][j],simp[i][v[1]]])
            new_simp.append(new_s)            

        for simplex in new_simp:
            for v in simplex:
                plt.plot(pts[v, 0], pts[v, 1], 'k-', color=color)
                
                
                
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


class PolyNet(nn.Module):
    def __init__(self, poly):
        super().__init__()

        self.net = convert_polynomial_to_net(poly)

    def forward(self, x):
        o = self.net.forward(x)
        return o


def convert_monomial_to_net(monom):
    bias = monom.coef[0].val
    weight = [x.val for x in monom.coef[1:]]

    layer = nn.Linear(len(weight), 1)

    layer.weight.data.copy_(to_tensor(weight))
    layer.bias.data.copy_(to_tensor(bias))

    return layer


def to_tensor(x):
    return torch.tensor(x).float()


def convert_polynomial_to_net(poly):
    if len(poly.monoms.keys()) == 1:
        c = list(poly.monoms.keys())[0]
        return convert_monomial_to_net(poly.monoms[c])

    weights = []
    biases = []
    for c in poly.monoms.keys():
        ab = poly.monoms[c].coef[0].val
        aw = [x.val for x in poly.monoms[c].coef[1:]]
        weights.append(to_tensor(aw))
        biases.append(to_tensor(ab))

    n = len(weights)
    layers = []

    if n % 2 == 0:
        k = 3 * (n // 2)

        basis = np.array([[1., -1.], [0., 1.], [0., -1.]])
        affine = to_tensor(np.kron(np.eye(n // 2, dtype=int), basis))
    else:
        k = 3 * (n // 2) - 1

        basis = np.array([[1., -1.], [0., 1.], [0., -1.]])
        affine = np.kron(np.eye((n - 1) // 2, dtype=int), basis)
        affine = np.vstack([affine, np.zeros(affine.shape[1])])
        affine = np.vstack([affine, np.zeros(affine.shape[1])])
        affine = np.hstack([affine, np.zeros((affine.shape[0], 1))])
        affine[-2, -1] = 1
        affine[-1, -1] = -1
        affine = to_tensor(affine)

    w = torch.stack(weights)
    b = torch.stack(biases)

    w = affine @ w
    b = affine @ b

    l0 = nn.Linear(*w.shape[::-1])

    l0.weight.data.copy_(w)
    l0.bias.data.copy_(b)

    layers.append(l0)
    layers.append(nn.ReLU())

    up = int(np.ceil(np.log2(n)))
    for i in range(1, up):

        l_of = layers[-2].out_features
        if l_of % 3 == 0:
            l_of = l_of // 3
            basis = np.array([[1., 1., -1.]])
            w = to_tensor(np.kron(np.eye(l_of, dtype=int), basis))
            b = torch.zeros(len(w))
        else:
            basis = np.array([[1., 1., -1.]])
            l_of = l_of // 3
            w = to_tensor(np.kron(np.eye(l_of, dtype=int), basis))
            w = np.vstack([w, np.zeros(w.shape[1])])
            w = np.hstack([w, np.zeros((w.shape[0], 1))])
            w = np.hstack([w, np.zeros((w.shape[0], 1))])
            w[-1, -2] = 1
            w[-1, -1] = -1
            w = to_tensor(w)
            b = torch.zeros(len(w))

        if w.shape[0] % 2 == 0:

            k = w.shape[0]

            basis = np.array([[1., -1.], [0., 1.], [0., -1.]])
            affine = to_tensor(np.kron(np.eye(k // 2, dtype=int), basis))

        else:
            k = w.shape[0]

            basis = np.array([[1., -1.], [0., 1.], [0., -1.]])
            affine = to_tensor(np.kron(np.eye((k - 1) // 2, dtype=int), basis))
            affine = np.vstack([affine, np.zeros(affine.shape[1])])
            affine = np.vstack([affine, np.zeros(affine.shape[1])])
            affine = np.hstack([affine, np.zeros((affine.shape[0], 1))])
            affine[-2, -1] = 1
            affine[-1, -1] = -1
            affine = to_tensor(affine)

        w = affine @ w
        b = affine @ b

        l = nn.Linear(*w.shape[::-1])

        l.weight.data.copy_(w)
        l.bias.data.copy_(b)

        layers.append(l)
        layers.append(nn.ReLU())

    l2 = nn.Linear(3, 1)
    l2.weight.data.copy_(torch.tensor([1., 1., -1.]))
    l2.bias.data.copy_(torch.tensor([0.]))

    layers.append(l2)

    net = nn.Sequential(*layers)

    return net


class DiffPolyNet(nn.Module):
    def __init__(self, poly1, poly2):
        super().__init__()

        self.net = convert_polynomial_diff_to_net(poly1, poly2)

    def forward(self, x):
        x = torch.cat((x, x), dim=0)
        o = self.net.forward(x)
        return o


def convert_polynomial_diff_to_net(poly1, poly2):
    net1 = convert_polynomial_to_net(poly1)
    net2 = convert_polynomial_to_net(poly2)

    layers = []
    flag = False

    try:
        len_net1 = len(net1)
    except:
        len_net1 = 1
        net1 = [net2]

    try:
        len_net2 = len(net2)
    except:
        len_net2 = 1
        net2 = [net2]

    n = max(len_net1, len_net2)
    for i in range(0, n, 2):
        if (i < len_net1) & (i < len_net2):

            w1 = net1[i].weight.data
            b1 = net1[i].bias.data

            w2 = net2[i].weight.data
            b2 = net2[i].bias.data

            r_1 = torch.cat((w1, torch.zeros((w1.shape[0], w2.shape[1]))), dim=1)
            r_2 = torch.cat((torch.zeros((w2.shape[0], w1.shape[1])), w2), dim=1)

            w = torch.cat((r_1, r_2), dim=0)

            b = torch.cat((b1, b2), dim=0)

            l = nn.Linear(*w.shape[::-1])

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)

        elif i < len_net1:

            w = net1[i].weight.data
            b = net1[i].bias.data

            if not flag:
                affine = torch.zeros((w.shape[1], w.shape[1] + 1))
                for j in range(affine.shape[0] - 1):
                    affine[j, j] = 1
                    if j % 3 == 0:
                        affine[j, -1] = -1

                w = w @ affine

            l = nn.Linear(*w.shape[::-1])

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)

            flag = True

        elif i < len_net2:
            w = net2[i].weight.data
            b = net2[i].bias.data

            if not flag:
                affine = torch.zeros((w.shape[1], w.shape[1] + 1))
                for j in range(affine.shape[0]):
                    affine[j, j + 1] = 1
                    if j % 3 == 0:
                        affine[j, 0] = -1
                w = w @ affine

            l = nn.Linear(*w.shape[::-1])
            if l.out_features == 1:
                w *= -1

            l.weight.data.copy_(w)
            l.bias.data.copy_(b)
            layers.append(l)
            flag = True

    net = [layers[0]]

    for j in range(1, len(layers)):
        net.append(nn.ReLU())
        net.append(layers[j])

    if net[-1].out_features == 2:
        la = nn.Linear(2, 1)
        la.weight.data.copy_(torch.tensor([1., -1.]))
        la.bias.data.copy_(torch.tensor([0.]))
        net.append(la)

    net = nn.Sequential(*net)

    return net
