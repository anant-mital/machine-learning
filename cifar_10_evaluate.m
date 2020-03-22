function [ out ] = cifar_10_evaluate( pred, gt )
  correct_matches = (pred == gt);
  correct_match_count = length(correct_matches(correct_matches==1));
  out = correct_match_count/length(gt);
end

