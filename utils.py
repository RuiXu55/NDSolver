def read_param(args):
  param = {}
  if not args.input :
    args.input = 'inp/input'
  with open(args.input,'r') as f:
    for line in f:
      if not line.isspace():
        data = line.split()
        if data[0] != '<':   
          key = data[0]
          val = float(data[2])
          if key in param.keys():
            param[key].append(val)
          else:
            param[key] = [] 
            param[key].append(val)
  print ("All user defined parameters: %s\n", param)
  return param
