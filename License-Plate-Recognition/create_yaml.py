import yaml
plate_yaml = dict(
    train ='/kaggle/working/datasets/train',
    val ='/kaggle/working/datasets/valid',
    test= '/kaggle/working/datasets/test',
    nc =27, #  number of classes"
    names = Names2, # class names
)

with open('plate.yaml', 'w') as outfile:
    yaml.dump(plate_yaml, outfile, default_flow_style=True)

# view the contents of the YAML file
%cat plate.yaml
