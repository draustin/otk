# d = np.choose(index, [i.d for i in isdbs])
# surface = np.choose(index, [i.surface for i in isdbs])
# face = np.choose(index, [i.face for i in isdbs])
# return ISDB(d, surface, face))
#
#
# # unionop
# @which.register
# def which(self:UnionOp, isdbs: Sequence[ISDB]) -> int:
#     d0 = isdbs[0].d
#     if np.isscalar(d0):
#         index0 = 0
#     else:
#         index0 = np.zeros(d0.shape, int)
#     for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
#         if np.isscalar(d0):
#             if isdb.d < d0:
#                 d0 = isdb.d
#                 index0 = index
#         else:
#             lt = isdb.d < d0
#             d0[lt] = isdb.d[lt]
#             index0[lt] = index
#     return index0
#
# @getsdb.register
# def getsdb(self:IntersectionOp, x):
#     d = self.surfaces[0].getsdb(x)
#     for surface in self.surfaces[1:]:
#         d = np.maximum(d, surface.getsdb(x))
#     return d
#
#
#
# @which.register
# def which(self:IntersectionOp, isdbs: Sequence[ISDB]) -> np.ndarray:
#     d0 = isdbs[0].d
#     index0 = np.zeros(d0.shape, int)
#     for index, isdb in zip(range(1, len(isdbs)), isdbs[1:]):
#         lt = isdb.d < d0
#         d0[lt] = isdb.d[lt]
#         index0[lt] = index
#     return index0