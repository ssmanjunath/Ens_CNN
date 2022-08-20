# Exponential moving average implemented in shadow copy
class EMA:
   
    def __init__(self,model,decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        for name, param in model.named_parameters():
            if param.requires_grad():
                self.shadow[name] = param.data.clone()

    def register(self,model,updates):
        decay = min(self.decay,(1.0 + updates)/(10.0 + updates))
        for name , param in model.named_parameters():
            if param.requires_grad():
                assert name in self.shadow
                new_average= (1-decay)*param.data+decay*self.shadow[name]
                self.shadow[name] = new_average.clone()
  
    def assign(self,model):
        for name, param in model.named_parmeters():
            if param.requires_grad():
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self,model):
        for name, param in model.named_parameters():
            if param.requires_grad():
                assert name in self.shadow
                param.data = self.original[name]
        