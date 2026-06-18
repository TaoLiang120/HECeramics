import os
import numpy as np
from pymatgen.core.composition import Composition
from heceramics.myglobal import VERY_SMALL_VALUE, Element_negativity, significant_figure4composition
from heceramics.myelements.myelements import get_DF_Index
from heceramics.utils.prettyformula import PrettyFormula


class CompstrUtil:
    def __init__(self, compstr, normalization=False, significant_figure=significant_figure4composition):
        '''
        Args:
            compstr: a composition string
            normalization: if parentheses present, normalize the composition inside parentheses.
                           if parentheses are already using pymatgen notation, normalization must be False.
        '''
        self.compstr = compstr
        self.normalization = normalization
        self.pretty_formula = self.get_pretty_formula(normalization=normalization, significant_figure=significant_figure)

    def get_pretty_formula(self, compstr=None, normalization=False, significant_figure=significant_figure4composition):
        if compstr is None: compstr = self.compstr
        pretty_formula = PrettyFormula.get(compstr,
                                                normalization=normalization, significant_figure=significant_figure)
        return pretty_formula

    def compstr2ROM(self, key, DF_REFERENCE, compstr=None, Index_style=0):
        if compstr is None: compstr = self.pretty_formula
        comp = Composition(compstr)
        nele = len(comp.elements)
        thisv = 0.0
        for i in range(nele):
            iele = comp.elements[i]
            isym = iele.symbol
            ifrac = comp.get_atomic_fraction(iele)
            index4DF = get_DF_Index(Index_style, isym)
            ival = DF_REFERENCE.loc[index4DF, key]
            thisv += ival * ifrac
        return thisv

    def compstr2delta(self, key, mean, DF_REFERENCE, compstr=None, Index_style=0):
        if compstr is None: compstr = self.pretty_formula
        comp = Composition(compstr)
        nele = len(comp.elements)
        thisv = 0.0
        for i in range(nele):
            iele = comp.elements[i]
            isym = iele.symbol
            ifrac = comp.get_atomic_fraction(iele)
            index4DF = get_DF_Index(Index_style, isym)
            ival = DF_REFERENCE.loc[index4DF, key]
            thisv += ifrac * np.power(1.0 - ival / mean, 2)
        return np.sqrt(thisv)

    def compstr2distort(self, key, DF_REFERENCE, compstr=None, Index_style=0):
        if compstr is None: compstr = self.pretty_formula
        comp = Composition(compstr)
        nele = len(comp.elements)
        thisv = 0.0
        thismax = 0.0
        for i in range(nele):
            iele = comp.elements[i]
            isym = iele.symbol
            ifrac = comp.get_atomic_fraction(iele)
            index4DF_i = get_DF_Index(Index_style, isym)
            ival = DF_REFERENCE.loc[index4DF_i, key]
            ival_tot = 0.0
            for j in range(nele):
                jele = comp.elements[j]
                jsym = jele.symbol
                jfrac = comp.get_atomic_fraction(jele)
                index4DF_j = get_DF_Index(Index_style, jsym)
                jval = DF_REFERENCE.loc[index4DF_j, key]
                v = jfrac * np.power((ival - jval)/(ival + jval), 2)
                ival_tot += v
            ival_tot = np.sqrt(ival_tot)
            ival_tot = ival_tot * ifrac * 9.0 / 8.0
            if ival_tot > thismax: thismax = ival_tot
            thisv += ival_tot
        return thisv, thismax


    def compstr2conc(self, compstr=None):
        if compstr is None: compstr = self.pretty_formula
        comp = Composition(compstr)
        nele = len(comp.elements)
        concs = np.zeros(nele)
        for i in range(len(comp.elements)):
            el = comp.elements[i]
            ifrac = comp.get_atomic_fraction(el)
            concs[i] = ifrac
        return np.array(concs)


    def compstr2sequential_conc(self, elements=Element_negativity, compstr=None):
        if compstr is None: compstr = self.pretty_formula
        comp = Composition(compstr)
        nele = len(elements)
        concs = np.zeros(nele)
        for i in range(len(comp.elements)):
            el = comp.elements[i]
            ind = elements.index(el.symbol)
            ifrac = comp.get_atomic_fraction(el)
            concs[ind] = ifrac
        return np.array(concs)

    @staticmethod
    def compstr2ROM_binary(compstrA, compstrB, key, DF_REFERENCE, Index_style=1):

        Acomp = Composition(compstrA)
        Bcomp = Composition(compstrB)
        nAele = len(Acomp.elements)
        nBele = len(Bcomp.elements)

        thisv = 0.0
        for i in range(nAele):
            iele = Acomp.elements[i]
            isym = iele.symbol
            ifrac =Acomp.get_atomic_fraction(iele)
            for j in range(nBele):
                jele = Bcomp.elements[j]
                jsym = jele.symbol
                jfrac = Bcomp.get_atomic_fraction(jele)
                index4DF = get_DF_Index(Index_style, isym, b=jsym)
                ival = DF_REFERENCE.loc[index4DF, key]
                thisv += ival * ifrac * jfrac
        return thisv



