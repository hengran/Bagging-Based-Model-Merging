import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig
from safetensors.torch import load_file, save_file
import json

def smart_key_mapping(adapter_key, base_keys):
    """
    智能匹配adapter键名到base model键名
    """
    # 移除lora相关后缀
    clean_key = adapter_key.replace('.lora_A.weight', '.weight')
    clean_key = clean_key.replace('.lora_B.weight', '.weight')
    clean_key = clean_key.replace('.lora_A.default.weight', '.weight')
    clean_key = clean_key.replace('.lora_B.default.weight', '.weight')
    
    # 尝试不同的前缀变体
    possible_keys = [
        clean_key,
        clean_key.replace('base_model.model.', ''),  # 移除base_model.model前缀
        clean_key.replace('base_model.', ''),        # 移除base_model前缀
        clean_key.replace('model.', '', 1),          # 移除第一个model前缀
    ]
    
    # 找到匹配的键
    for pk in possible_keys:
        if pk in base_keys:
            return pk
    
    return None

def manual_merge_lora_smart(base_model_path, peft_model_path, output_path):
    """
    智能键名映射的手动合并
    """
    print("="*60)
    print("🔧 智能LoRA合并（自动键名映射）")
    print("="*60)
    
    # 1. 加载配置
    config = PeftConfig.from_pretrained(peft_model_path)
    print(f"LoRA配置: r={config.r}, alpha={config.lora_alpha}")
    scaling = config.lora_alpha / config.r
    print(f"Scaling factor: {scaling}\n")
    
    # 2. 加载基础模型
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    base_state_dict = base_model.state_dict()
    base_keys = set(base_state_dict.keys())
    
    print(f"基础模型包含 {len(base_keys)} 个参数")
    print(f"基础模型键名示例:")
    for k in list(base_keys)[:3]:
        print(f"  {k}")
    
    # 3. 加载adapter
    print("\n加载LoRA adapter...")
    adapter_path = os.path.join(peft_model_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        adapter_weights = load_file(adapter_path)
    else:
        adapter_weights = torch.load(
            os.path.join(peft_model_path, "adapter_model.bin"),
            map_location='cpu'
        )
    
    print(f"Adapter包含 {len(adapter_weights)} 个参数")
    print(f"Adapter键名示例:")
    for k in list(adapter_weights.keys())[:3]:
        print(f"  {k}")
    
    # 4. 配对lora_A和lora_B
    print("\n配对LoRA权重...")
    lora_pairs = {}
    
    for key in adapter_weights.keys():
        if 'lora_A' in key:
            # 找到对应的base key
            base_key = smart_key_mapping(key, base_keys)
            if base_key:
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['lora_A'] = key
            else:
                print(f"⚠️ 未找到匹配: {key}")
        
        elif 'lora_B' in key:
            base_key = smart_key_mapping(key, base_keys)
            if base_key:
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['lora_B'] = key
            else:
                print(f"⚠️ 未找到匹配: {key}")
    
    print(f"\n找到 {len(lora_pairs)} 对LoRA权重")
    
    # 显示映射示例
    print("\n键名映射示例（前3个）:")
    for i, (base_key, lora_keys) in enumerate(list(lora_pairs.items())[:3]):
        print(f"  Base key: {base_key}")
        print(f"    lora_A: {lora_keys.get('lora_A', 'MISSING')}")
        print(f"    lora_B: {lora_keys.get('lora_B', 'MISSING')}")
    
    # 5. 合并权重
    print("\n开始合并...")
    merged_count = 0
    skipped_count = 0
    
    for base_key, lora_keys in lora_pairs.items():
        if 'lora_A' not in lora_keys or 'lora_B' not in lora_keys:
            print(f"⚠️ 跳过不完整的对: {base_key}")
            skipped_count += 1
            continue
        
        # 获取权重
        W_base = base_state_dict[base_key]
        lora_A = adapter_weights[lora_keys['lora_A']]
        lora_B = adapter_weights[lora_keys['lora_B']]
        
        # 确保在同一设备上
        lora_A = lora_A.to(W_base.device).to(W_base.dtype)
        lora_B = lora_B.to(W_base.device).to(W_base.dtype)
        
        # 计算增量: delta_W = (lora_B @ lora_A) * scaling
        delta_W = (lora_B @ lora_A) * scaling
        
        # 合并
        W_merged = W_base + delta_W
        base_state_dict[base_key] = W_merged
        
        merged_count += 1
        
        # 打印统计（每10个打印一次）
        if merged_count % 10 == 1:
            delta_norm = delta_W.norm().item()
            base_norm = W_base.norm().item()
            ratio = delta_norm / base_norm if base_norm > 0 else 0
            print(f"✓ [{merged_count}] {base_key}")
            print(f"  Delta: {delta_norm:.6e}, Base: {base_norm:.6e}, Ratio: {ratio:.6e}")
    
    print(f"\n✅ 成功合并: {merged_count} 个")
    print(f"⚠️ 跳过: {skipped_count} 个")
    
    if merged_count == 0:
        print("\n❌ 没有成功合并任何权重!")
        print("请检查上面的键名映射示例，可能需要手动调整映射规则。")
        return None
    
    # 6. 加载合并后的权重
    print("\n加载合并后的权重到模型...")
    base_model.load_state_dict(base_state_dict)
    
    # 7. 保存
    print(f"保存到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(output_path, safe_serialization=True)
    
    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ 保存完成!\n")
    return base_model

def verify_merge(base_model_path, merged_model_path):
    """
    验证合并
    """
    print("="*60)
    print("🔍 验证合并结果")
    print("="*60)
    
    print("加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    print("加载合并模型...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    base_state = base_model.state_dict()
    merged_state = merged_model.state_dict()
    
    diff_count = 0
    total_count = 0
    
    print("\n检查权重差异...")
    for name in base_state.keys():
        if name not in merged_state:
            continue
        
        total_count += 1
        base_param = base_state[name]
        merged_param = merged_state[name]
        
        max_diff = torch.abs(base_param - merged_param).max().item()
        
        if max_diff > 1e-7:
            diff_count += 1
            if diff_count <= 5:  # 只打印前5个
                mean_diff = torch.abs(base_param - merged_param).mean().item()
                print(f"✓ {name}")
                print(f"  Max: {max_diff:.6e}, Mean: {mean_diff:.6e}")
    
    print(f"\n{'='*60}")
    print(f"总参数: {total_count}")
    print(f"有差异: {diff_count}")
    print(f"差异率: {diff_count/total_count*100:.2f}%")
    print(f"{'='*60}\n")
    
    return diff_count > 0

def inference_compare(base_model_path, merged_model_path):
    """
    推理对比
    """
    print("="*60)
    print("🧪 推理对比测试")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print("加载模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # 测试
    test_texts = [
        "The capital of France is",
        "What is the meaning of life?",
    ]
    
    has_difference = False
    
    for text in test_texts:
        print(f"\n📝 输入: '{text}'")
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            base_out = base_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            merged_out = merged_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        merged_text = tokenizer.decode(merged_out[0], skip_special_tokens=True)
        
        print(f"  Base:   {base_text}")
        print(f"  Merged: {merged_text}")
        
        if base_text != merged_text:
            print("  ✅ 不同!")
            has_difference = True
        else:
            print("  ⚠️ 相同")
    
    return has_difference

# ============================================
# 主流程
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并Qwen模型和LoRA参数")
    parser.add_argument("--base_model", type=str, required=True, help="原始模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--output_path", type=str, default="./merged_model", help="合并后模型保存路径")
    args = parser.parse_args()
    
    base_model_path = args.base_model
    peft_model_path = args.lora_path
    output_path = args.output_path
    
    # 执行合并
    merged_model = manual_merge_lora_smart(base_model_path, peft_model_path, output_path)
    
    if merged_model is None:
        print("\n❌ 合并失败，退出")
        exit(1)
    
    # 验证
    if verify_merge(base_model_path, output_path):
        print("✅ 权重验证通过!")
        
        # 推理测试
    #     if inference_compare(base_model_path, output_path):
    #         print("\n🎉🎉🎉 完全成功! 合并后的模型已保存到:", output_path)
    #     else:
    #         print("\n⚠️ 权重已合并但推理结果相同（可能LoRA效果很小）")
    else:
        print("❌ 验证失败")
    # base_model_path = "/mnt/data/models/Qwen/Qwen3-4B"
    # peft_model_path = "model_merge_checkpoint/Qwen3-4B-imdb_classification_0.02_2_1e-4_32_lora32"
    # output_path = "merged_model_imdb"
    
    