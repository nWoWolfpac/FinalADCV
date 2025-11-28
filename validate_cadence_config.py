#!/usr/bin/env python3
"""
Validate Cadence YAML configuration files
Usage: python validate_cadence_config.py [config_file.yaml]
"""

import yaml
import sys
from pathlib import Path


def validate_yaml_syntax(yaml_path):
    """Check if YAML file has valid syntax"""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Valid YAML syntax: {yaml_path}")
        return config
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML syntax in {yaml_path}:")
        print(f"   {e}")
        return None
    except FileNotFoundError:
        print(f"❌ File not found: {yaml_path}")
        return None


def validate_cadence_config(config, yaml_path):
    """Validate Cadence-specific configuration"""
    errors = []
    warnings = []
    
    # Check required fields
    if 'name' not in config:
        errors.append("Missing required field: 'name'")
    
    if 'script' not in config:
        errors.append("Missing required field: 'script'")
    else:
        # Check if script exists
        script_path = Path(yaml_path).parent.parent / config['script']
        if not script_path.exists():
            errors.append(f"Script not found: {config['script']}")
    
    # Check resources
    if 'resources' in config:
        resources = config['resources']
        
        # Validate GPU
        if 'gpu' in resources:
            valid_gpus = ['T4', 'A10', 'A100', 'none']
            if resources['gpu'] not in valid_gpus:
                errors.append(f"Invalid GPU: {resources['gpu']}. Must be one of {valid_gpus}")
        
        # Validate memory
        if 'memory' in resources:
            memory = resources['memory']
            if not isinstance(memory, str) or not memory.endswith('GB'):
                warnings.append(f"Memory should be specified as string with 'GB' suffix (e.g., '32GB')")
        
        # Validate timeout
        if 'timeout' in resources:
            timeout = resources['timeout']
            if not isinstance(timeout, str):
                warnings.append(f"Timeout should be specified as string (e.g., '6h', '30m')")
    
    # Check environment
    if 'environment' in config:
        env = config['environment']
        
        # Check requirements file
        if 'requirements' in env:
            req_path = Path(yaml_path).parent.parent / env['requirements']
            if not req_path.exists():
                warnings.append(f"Requirements file not found: {env['requirements']}")
    
    # Check cost control
    if 'cost' in config:
        cost = config['cost']
        if 'max_cost' in cost:
            if not isinstance(cost['max_cost'], (int, float)):
                errors.append("max_cost must be a number")
    
    # Print results
    if errors:
        print(f"\n❌ Validation failed for {yaml_path}:")
        for error in errors:
            print(f"   ERROR: {error}")
    
    if warnings:
        print(f"\n⚠️  Warnings for {yaml_path}:")
        for warning in warnings:
            print(f"   WARNING: {warning}")
    
    if not errors and not warnings:
        print(f"✅ Configuration is valid: {yaml_path}")
    
    return len(errors) == 0


def print_config_summary(config):
    """Print a summary of the configuration"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Name:        {config.get('name', 'N/A')}")
    print(f"Script:      {config.get('script', 'N/A')}")
    
    if 'args' in config:
        print(f"Arguments:")
        for key, value in config['args'].items():
            print(f"  --{key}: {value}")
    
    if 'resources' in config:
        res = config['resources']
        print(f"\nResources:")
        print(f"  GPU:     {res.get('gpu', 'N/A')}")
        print(f"  Memory:  {res.get('memory', 'N/A')}")
        print(f"  Timeout: {res.get('timeout', 'N/A')}")
    
    if 'cost' in config:
        cost = config['cost']
        print(f"\nCost Control:")
        print(f"  Max Cost:   ${cost.get('max_cost', 'N/A')}")
        print(f"  Auto Stop:  {cost.get('auto_stop', 'N/A')}")
    
    print("="*60)


def main():
    if len(sys.argv) > 1:
        # Validate specific file
        yaml_path = Path(sys.argv[1])
        config = validate_yaml_syntax(yaml_path)
        if config:
            if validate_cadence_config(config, yaml_path):
                print_config_summary(config)
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    else:
        # Validate all configs in cadence_configs/
        configs_dir = Path(__file__).parent / "cadence_configs"
        
        if not configs_dir.exists():
            print(f"❌ Directory not found: {configs_dir}")
            sys.exit(1)
        
        yaml_files = list(configs_dir.glob("*.yaml"))
        
        if not yaml_files:
            print(f"❌ No YAML files found in {configs_dir}")
            sys.exit(1)
        
        print(f"Found {len(yaml_files)} YAML config files\n")
        
        all_valid = True
        for yaml_file in yaml_files:
            config = validate_yaml_syntax(yaml_file)
            if config:
                if not validate_cadence_config(config, yaml_file):
                    all_valid = False
            else:
                all_valid = False
            print()  # Empty line between files
        
        if all_valid:
            print("\n✅ All configurations are valid!")
            sys.exit(0)
        else:
            print("\n❌ Some configurations have errors")
            sys.exit(1)


if __name__ == "__main__":
    main()
