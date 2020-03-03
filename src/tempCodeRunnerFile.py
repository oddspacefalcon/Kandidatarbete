sum_Ps_s = torch.sum(self.Ps[s]) 
self.Ps[s] = self.Ps[s]/sum_Ps_s 