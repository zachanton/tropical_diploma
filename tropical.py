import math
import numpy as np

class Tropical:
    
    def __init__(self,val):
        self.val = val
    
    # Relations
    def __lt__(self,other):
        if type(other) == Tropical:
            return self.val-other.val < 0
        else:
            return self.val-other < 0
    
    def __gt__(self,other):
        if type(other) == Tropical:
            return self.val-other.val > 0
        else:
            return self.val-other > 0
    
    def __le__(self,other):
        if type(other) == Tropical:
            return self.val-other.val <= 0
        else:
            return self.val-other <= 0
    
    def __ge__(self,other):
        if type(other) == Tropical:
            return self.val-other.val >= 0
        else:
            return self.val-other >= 0
    
    def __eq__(self,other):
        if type(other) == Tropical:
            return self.val == other.val
        else:
            return self.val == other
    
    
    # Simple operations
    def __add__(self,other):
        if type(other) == Tropical:
            return Tropical(max(self.val,other.val))
        else:
            return Tropical(max(self.val,other))
        
    def __radd__(self,other):
        if type(other) == Tropical:
            return Tropical(max(self.val,other.val))
        else:
            return Tropical(max(self.val,other))
    
    def __mul__(self,other):
        if type(other) == Tropical:
            return Tropical(self.val+other.val)
        else:
            return Tropical(self.val+other)
        
    def __rmul__(self,other):
        if type(other) == Tropical:
            return Tropical(self.val+other.val)
        else:
            return Tropical(self.val+other)
        
    def __pow__(self,other):
        if type(other) == Tropical:
            assert float(other)==int(float(other)), 'pow should be natural'
            assert float(other)>=0, 'pow should be natural'
            return Tropical(self.val*other.val)
        else:
            assert float(other)==int(float(other)), 'pow should be natural'
            assert float(other)>=0, 'pow should be natural'
            return Tropical(self.val*other)
        
        

    # Otheer
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
    
    
    def __truediv__(self,b):
        return self * b.sym()
    

    def __floordiv__(self,b):
        return self * b.sym()
    
    
import string
class TropicalMonomial:
    
    def __init__(self,coef):
        if type(coef)==TropicalMonomial:
            self.coef = coef.coef
        else:
            self.coef = [Tropical(x)  if not isinstance(x, Tropical) else x for x in coef]
        
    def __getitem__(self,n):
        return self.coef[n]
    
    def __len__(self):
        return len(self.coef)
    
    def __eq__(self,mono):
        if len(self) == len(mono):
            if all([x == y for x,y in zip(self.coef,mono.coef)]):
                return True
        return False
    
    
    
    def __add__(self,other):
        assert type(other) == TropicalMonomial
        assert len(self)==len(other)
        
        return TropicalPolynomial([self.coef, other.coef])
    
    def __radd__(self,other):
        assert type(other) == TropicalMonomial
        assert len(self)==len(other)
        
        return TropicalPolynomial([self.coef, other.coef])
    
    
    def __mul__(self,other):
        if type(other) == TropicalMonomial:
            assert len(self)==len(other)

            return TropicalMonomial([x * y for x,y in zip(self.coef,other.coef)])
        
        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i==0 else x  for i,x in enumerate(self.coef)])
    
    def __rmul__(self,other):
        if type(other) == TropicalMonomial:
            assert len(self)==len(other)

            return TropicalMonomial([x * y for x,y in zip(self.coef,other.coef)])
        elif type(other) == Tropical:
            return TropicalMonomial([x * other if i==0 else x  for i,x in enumerate(self.coef)])
    
    
    
    def __div__(self,other):
        assert type(other) == TropicalMonomial
        assert len(self)==len(other)
        
        return TropicalMonomial([x / y for x,y in zip(self.coef,other.coef)])
    
    def __rdiv__(self,other):
        assert type(other) == TropicalMonomial
        assert len(self)==len(other)
        
        return TropicalMonomial([x / y for x,y in zip(self.coef,other.coef)])

    def __pow__(self,other):
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
            
            v = var[pwr-1]
            
            if coe == 0:
                continue
            
            if pwr == 0:
                s = '{}'.format(coe)
                
            else:
                if coe == 1:
                    s = '{}'.format(v)
                    
                else:
                    s = '{}^{}'.format(v, coe)

            out.append(s)
        
        out = '⨀'.join(out)
        
        return out
    
    
    def evaluate(self,X):
        """Evaluate the monomial at a given point or points"""
        X = [Tropical(x) for x in X]
        assert len(X)==len(self)-1
            
        out = self.coef[0]
        
        for pwr,coef in enumerate(self.coef[1:]):
            out *= X[pwr]**coef

        return out
    
class TropicalPolynomial:
    
    def __init__(self,monoms):
        
        if isinstance(monoms, list):
            self.monoms = {}

            for x in monoms:
                if any(isinstance(i,Tropical) for i in x):
                    x = [i.val for i in x]
                
                if tuple(x[1:]) in self.monoms.keys():
                    prev_mon = self.monoms[tuple(x[1:])]
                    new_x = [prev_mon[0]+x[0]]+list(x[1:])
                    self.monoms[tuple(new_x[1:])] = TropicalMonomial(new_x)
                else:
                    self.monoms[tuple(x[1:])] = TropicalMonomial(x)
        elif isinstance(monoms,dict):
            self.monoms = monoms
        
    def __getitem__(self,n):
        return list(self.monoms.values())[n]
    
    def __len__(self):
        return len(self.monoms)
    
    def __eq__(self,other):
        if len(set(self.monoms.keys()).symmetric_difference(other.monoms.keys()))==0:
            return True
        return False
    
    def __add__(self,other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0]+new_monom[x][0]]+list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return self+other
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val]+[0 for _ in range(len(self[0].coef)-1)]])
            return self+other            
        return TropicalPolynomial(new_monom)
        
    def __radd__(self,other):
        new_monom = self.monoms.copy()
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                if x in new_monom.keys():
                    new_x = [other.monoms[x][0]+new_monom[x][0]]+list(new_monom[x][1:])
                    new_monom[x] = TropicalMonomial(new_x)
                else:
                    new_monom[x] = other.monoms[x]
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return self+other
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val]+[0 for _ in range(len(self[0].coef)-1)]])
            return self+other            
                    
        return TropicalPolynomial(new_monom)
    
    
    def __mul__(self,other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    new_monom[key_z] = z
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return self*other
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val]+[0 for _ in range(len(self[0].coef)-1)]])
            return self*other
                    
        return TropicalPolynomial(new_monom)
    
    def __rmul__(self,other):
        new_monom = {}
        if isinstance(other, TropicalPolynomial):
            for x in other.monoms:
                for y in self.monoms:
                    z = other.monoms[x] * self.monoms[y]
                    key_z = tuple([i.val for i in z.coef[1:]])
                    new_monom[key_z] = z
        elif isinstance(other, TropicalMonomial):
            other = TropicalPolynomial([other.coef])
            return self*other
        elif isinstance(other, Tropical):
            other = TropicalPolynomial([[other.val]+[0 for _ in range(len(self[0].coef)-1)]])
            return self*other
                    
        return TropicalPolynomial(new_monom)
    
    def __pow__(self,other):
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
            
        out = ' ⨁ '.join(out)
        
        return out
    
    
    def evaluate(self,X):
        """Evaluate the polynomial at a given point or points"""
        out = []
        
        for x in self.monoms:
            out.append(self.monoms[x].evaluate(X))

        out = max(out)
        
        return out