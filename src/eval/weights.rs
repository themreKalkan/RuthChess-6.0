
use once_cell::sync::Lazy;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy)]
pub struct WeightSet {
    pub material_weight: f32,
    pub bishop_pair_weight: f32,
    pub minor_piece_adjustment_weight: f32,
    
    pub pst_weight: f32,
    
    pub pawn_structure_weight: f32,
    pub passed_pawn_weight: f32,
    pub isolated_pawn_weight: f32,
    pub doubled_pawn_weight: f32,
    pub backward_pawn_weight: f32,
    pub pawn_chain_weight: f32,
    
    pub mobility_weight: f32,
    pub knight_mobility_weight: f32,
    pub bishop_mobility_weight: f32,
    pub rook_mobility_weight: f32,
    pub queen_mobility_weight: f32,
    
    pub space_weight: f32,
    pub center_control_weight: f32,
    
    pub king_safety_weight: f32,
    pub king_shield_weight: f32,
    pub king_attack_weight: f32,
    pub king_zone_attack_weight: f32,
    
    pub imbalance_weight: f32,
    
    pub threats_weight: f32,
    pub hanging_piece_weight: f32,
    pub tactical_threat_weight: f32,
    
    pub tempo_bonus_weight: f32,
    
    pub endgame_scale_factor: f32,
}

impl Default for WeightSet {
    fn default() -> Self {
        Self {
            material_weight: 1.00,
            bishop_pair_weight: 1.15,
            minor_piece_adjustment_weight: 0.85,
            
            pst_weight: 0.92,
            
            pawn_structure_weight: 1.08,
            passed_pawn_weight: 1.35,
            isolated_pawn_weight: 1.10,    
            doubled_pawn_weight: 1.05,
            backward_pawn_weight: 1.00,
            pawn_chain_weight: 0.95,
            
            mobility_weight: 1.12,
            knight_mobility_weight: 1.00,
            bishop_mobility_weight: 1.10,
            rook_mobility_weight: 1.15,
            queen_mobility_weight: 0.85,
            
            space_weight: 0.75,
            center_control_weight: 0.88,
            
            king_safety_weight: 1.25,
            king_shield_weight: 1.15,
            king_attack_weight: 1.30,
            king_zone_attack_weight: 1.20,
            
            imbalance_weight: 0.65,
            
            threats_weight: 0.98,          
            hanging_piece_weight: 1.20,
            tactical_threat_weight: 1.05,
            
            tempo_bonus_weight: 1.10,
            
            endgame_scale_factor: 1.00,
        }
    }
}

pub static WEIGHTS: Lazy<RwLock<WeightSet>> = Lazy::new(|| {
    RwLock::new(WeightSet::default())
});

pub fn aggressive_weights() -> WeightSet {
    WeightSet {
        material_weight: 0.95,
        bishop_pair_weight: 1.20,
        minor_piece_adjustment_weight: 0.90,
        pst_weight: 1.00,
        pawn_structure_weight: 1.00,
        passed_pawn_weight: 1.40,
        isolated_pawn_weight: 0.95,
        doubled_pawn_weight: 0.95,
        backward_pawn_weight: 0.90,
        pawn_chain_weight: 1.10,
        mobility_weight: 1.25,
        knight_mobility_weight: 1.10,
        bishop_mobility_weight: 1.20,
        rook_mobility_weight: 1.25,
        queen_mobility_weight: 0.90,
        space_weight: 0.85,
        center_control_weight: 1.00,
        king_safety_weight: 1.35,
        king_shield_weight: 1.10,
        king_attack_weight: 1.45,
        king_zone_attack_weight: 1.35,
        imbalance_weight: 0.70,
        threats_weight: 1.15,
        hanging_piece_weight: 1.25,
        tactical_threat_weight: 1.20,
        tempo_bonus_weight: 1.20,
        endgame_scale_factor: 1.00,
    }
}

pub fn positional_weights() -> WeightSet {
    WeightSet {
        material_weight: 1.05,
        bishop_pair_weight: 1.25,
        minor_piece_adjustment_weight: 0.95,
        pst_weight: 1.05,
        pawn_structure_weight: 1.20,
        passed_pawn_weight: 1.30,
        isolated_pawn_weight: 1.20,
        doubled_pawn_weight: 1.15,
        backward_pawn_weight: 1.10,
        pawn_chain_weight: 1.00,
        mobility_weight: 1.00,
        knight_mobility_weight: 0.95,
        bishop_mobility_weight: 1.05,
        rook_mobility_weight: 1.10,
        queen_mobility_weight: 0.80,
        space_weight: 0.90,
        center_control_weight: 1.05,
        king_safety_weight: 1.15,
        king_shield_weight: 1.20,
        king_attack_weight: 1.10,
        king_zone_attack_weight: 1.05,
        imbalance_weight: 0.75,
        threats_weight: 0.85,
        hanging_piece_weight: 1.15,
        tactical_threat_weight: 0.90,
        tempo_bonus_weight: 0.95,
        endgame_scale_factor: 1.00,
    }
}

pub fn endgame_weights() -> WeightSet {
    WeightSet {
        material_weight: 1.10,
        bishop_pair_weight: 1.30,
        minor_piece_adjustment_weight: 1.00,
        pst_weight: 0.80,
        pawn_structure_weight: 1.25,
        passed_pawn_weight: 1.50,
        isolated_pawn_weight: 1.15,
        doubled_pawn_weight: 1.10,
        backward_pawn_weight: 1.05,
        pawn_chain_weight: 0.85,
        mobility_weight: 1.05,
        knight_mobility_weight: 0.90,
        bishop_mobility_weight: 1.00,
        rook_mobility_weight: 1.20,
        queen_mobility_weight: 0.75,
        space_weight: 0.60,
        center_control_weight: 0.70,
        king_safety_weight: 0.85,
        king_shield_weight: 0.70,
        king_attack_weight: 0.80,
        king_zone_attack_weight: 0.75,
        imbalance_weight: 0.80,
        threats_weight: 0.80,
        hanging_piece_weight: 1.10,
        tactical_threat_weight: 0.85,
        tempo_bonus_weight: 0.85,
        endgame_scale_factor: 1.10,
    }
}

pub fn get_weights() -> WeightSet {
    *WEIGHTS.read().unwrap()
}

pub fn set_weights(new_weights: WeightSet) {
    *WEIGHTS.write().unwrap() = new_weights;
}

pub fn update_weight<F>(updater: F) 
where
    F: FnOnce(&mut WeightSet),
{
    let mut weights = WEIGHTS.write().unwrap();
    updater(&mut *weights);
}

pub fn use_aggressive_style() {
    set_weights(aggressive_weights());
}

pub fn use_positional_style() {
    set_weights(positional_weights());
}

pub fn use_endgame_style() {
    set_weights(endgame_weights());
}

#[inline(always)]
pub fn material_weight() -> f32 {
    WEIGHTS.read().unwrap().material_weight
}

#[inline(always)]
pub fn bishop_pair_weight() -> f32 {
    WEIGHTS.read().unwrap().bishop_pair_weight
}

#[inline(always)]
pub fn pst_weight() -> f32 {
    WEIGHTS.read().unwrap().pst_weight
}

#[inline(always)]
pub fn pawn_structure_weight() -> f32 {
    WEIGHTS.read().unwrap().pawn_structure_weight
}

#[inline(always)]
pub fn mobility_weight() -> f32 {
    WEIGHTS.read().unwrap().mobility_weight
}

#[inline(always)]
pub fn space_weight() -> f32 {
    WEIGHTS.read().unwrap().space_weight
}

#[inline(always)]
pub fn king_safety_weight() -> f32 {
    WEIGHTS.read().unwrap().king_safety_weight
}

#[inline(always)]
pub fn imbalance_weight() -> f32 {
    WEIGHTS.read().unwrap().imbalance_weight
}

#[inline(always)]
pub fn threats_weight() -> f32 {
    WEIGHTS.read().unwrap().threats_weight
}

#[inline(always)]
pub fn tempo_bonus_weight() -> f32 {
    WEIGHTS.read().unwrap().tempo_bonus_weight
}

pub fn reset_weights() {
    set_weights(WeightSet::default());
}

pub fn scale_all_weights(factor: f32) {
    update_weight(|w| {
        w.material_weight *= factor;
        w.bishop_pair_weight *= factor;
        w.minor_piece_adjustment_weight *= factor;
        w.pst_weight *= factor;
        w.pawn_structure_weight *= factor;
        w.passed_pawn_weight *= factor;
        w.isolated_pawn_weight *= factor;
        w.doubled_pawn_weight *= factor;
        w.backward_pawn_weight *= factor;
        w.pawn_chain_weight *= factor;
        w.mobility_weight *= factor;
        w.knight_mobility_weight *= factor;
        w.bishop_mobility_weight *= factor;
        w.rook_mobility_weight *= factor;
        w.queen_mobility_weight *= factor;
        w.space_weight *= factor;
        w.center_control_weight *= factor;
        w.king_safety_weight *= factor;
        w.king_shield_weight *= factor;
        w.king_attack_weight *= factor;
        w.king_zone_attack_weight *= factor;
        w.imbalance_weight *= factor;
        w.threats_weight *= factor;
        w.hanging_piece_weight *= factor;
        w.tactical_threat_weight *= factor;
        w.tempo_bonus_weight *= factor;
        w.endgame_scale_factor *= factor;
    });
}

use crate::eval::evaluate::Score;

#[inline(always)]
pub fn apply_weight(score: Score, weight: f32) -> Score {
    Score::new(
        (score.mg as f32 * weight) as i32,
        (score.eg as f32 * weight) as i32,
    )
}

#[inline(always)]
pub fn apply_weight_i32(value: i32, weight: f32) -> i32 {
    (value as f32 * weight) as i32
}

pub fn parse_weights_from_string(input: &str) -> Result<WeightSet, String> {
    let mut weights = WeightSet::default();
    
    for line in input.lines() {
        let parts: Vec<&str> = line.split('=').collect();
        if parts.len() != 2 {
            continue;
        }
        
        let name = parts[0].trim();
        let value = parts[1].trim().parse::<f32>()
            .map_err(|_| format!("Invalid value for {}: {}", name, parts[1]))?;
        
        match name {
            "material" => weights.material_weight = value,
            "bishop_pair" => weights.bishop_pair_weight = value,
            "minor_piece_adjustment" => weights.minor_piece_adjustment_weight = value,
            "pst" => weights.pst_weight = value,
            "pawn_structure" => weights.pawn_structure_weight = value,
            "passed_pawn" => weights.passed_pawn_weight = value,
            "isolated_pawn" => weights.isolated_pawn_weight = value,
            "doubled_pawn" => weights.doubled_pawn_weight = value,
            "backward_pawn" => weights.backward_pawn_weight = value,
            "pawn_chain" => weights.pawn_chain_weight = value,
            "mobility" => weights.mobility_weight = value,
            "knight_mobility" => weights.knight_mobility_weight = value,
            "bishop_mobility" => weights.bishop_mobility_weight = value,
            "rook_mobility" => weights.rook_mobility_weight = value,
            "queen_mobility" => weights.queen_mobility_weight = value,
            "space" => weights.space_weight = value,
            "center_control" => weights.center_control_weight = value,
            "king_safety" => weights.king_safety_weight = value,
            "king_shield" => weights.king_shield_weight = value,
            "king_attack" => weights.king_attack_weight = value,
            "king_zone_attack" => weights.king_zone_attack_weight = value,
            "imbalance" => weights.imbalance_weight = value,
            "threats" => weights.threats_weight = value,
            "hanging_piece" => weights.hanging_piece_weight = value,
            "tactical_threat" => weights.tactical_threat_weight = value,
            "tempo_bonus" => weights.tempo_bonus_weight = value,
            "endgame_scale" => weights.endgame_scale_factor = value,
            _ => return Err(format!("Unknown weight: {}", name)),
        }
    }
    
    Ok(weights)
}

pub fn weights_to_string(weights: &WeightSet) -> String {
    format!(
        "material={:.2}\n\
        bishop_pair={:.2}\n\
        minor_piece_adjustment={:.2}\n\
        pst={:.2}\n\
        pawn_structure={:.2}\n\
        passed_pawn={:.2}\n\
        isolated_pawn={:.2}\n\
        doubled_pawn={:.2}\n\
        backward_pawn={:.2}\n\
        pawn_chain={:.2}\n\
        mobility={:.2}\n\
        knight_mobility={:.2}\n\
        bishop_mobility={:.2}\n\
        rook_mobility={:.2}\n\
        queen_mobility={:.2}\n\
        space={:.2}\n\
        center_control={:.2}\n\
        king_safety={:.2}\n\
        king_shield={:.2}\n\
        king_attack={:.2}\n\
        king_zone_attack={:.2}\n\
        imbalance={:.2}\n\
        threats={:.2}\n\
        hanging_piece={:.2}\n\
        tactical_threat={:.2}\n\
        tempo_bonus={:.2}\n\
        endgame_scale={:.2}",
        weights.material_weight,
        weights.bishop_pair_weight,
        weights.minor_piece_adjustment_weight,
        weights.pst_weight,
        weights.pawn_structure_weight,
        weights.passed_pawn_weight,
        weights.isolated_pawn_weight,
        weights.doubled_pawn_weight,
        weights.backward_pawn_weight,
        weights.pawn_chain_weight,
        weights.mobility_weight,
        weights.knight_mobility_weight,
        weights.bishop_mobility_weight,
        weights.rook_mobility_weight,
        weights.queen_mobility_weight,
        weights.space_weight,
        weights.center_control_weight,
        weights.king_safety_weight,
        weights.king_shield_weight,
        weights.king_attack_weight,
        weights.king_zone_attack_weight,
        weights.imbalance_weight,
        weights.threats_weight,
        weights.hanging_piece_weight,
        weights.tactical_threat_weight,
        weights.tempo_bonus_weight,
        weights.endgame_scale_factor,
    )
}

pub fn adjust_weights_for_phase(phase: u32) {
    let endgame_factor = 1.0 - (phase as f32 / 256.0);
    
    update_weight(|w| {
        w.pawn_structure_weight = 1.08 + 0.17 * endgame_factor;
        w.passed_pawn_weight = 1.35 + 0.25 * endgame_factor;
        
        w.king_safety_weight = 1.25 - 0.40 * endgame_factor;
        w.king_attack_weight = 1.30 - 0.50 * endgame_factor;
        
        w.rook_mobility_weight = 1.15 + 0.15 * endgame_factor;
        
        w.space_weight = 0.75 - 0.25 * endgame_factor;
        w.center_control_weight = 0.88 - 0.28 * endgame_factor;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_weights() {
        let weights = WeightSet::default();
        assert_eq!(weights.material_weight, 1.00);
        assert_eq!(weights.king_safety_weight, 1.25);
    }
    
    #[test]
    fn test_weight_presets() {
        let aggressive = aggressive_weights();
        assert!(aggressive.king_attack_weight > aggressive.king_shield_weight);
        
        let positional = positional_weights();
        assert!(positional.pawn_structure_weight > positional.threats_weight);
        
        let endgame = endgame_weights();
        assert!(endgame.passed_pawn_weight > endgame.king_safety_weight);
    }
    
    #[test]
    fn test_weight_update() {
        reset_weights();
        
        update_weight(|w| {
            w.material_weight = 1.5;
        });
        
        assert_eq!(material_weight(), 1.5);
        
        reset_weights();
        assert_eq!(material_weight(), 1.00);
    }
    
    #[test]
    fn test_apply_weight() {
        let score = Score::new(100, 50);
        let weighted = apply_weight(score, 1.5);
        
        assert_eq!(weighted.mg, 150);
        assert_eq!(weighted.eg, 75);
    }
    
    #[test]
    fn test_parse_weights() {
        let input = "material=1.2\npst=0.9\nking_safety=1.1";
        let weights = parse_weights_from_string(input).unwrap();
        
        assert_eq!(weights.material_weight, 1.2);
        assert_eq!(weights.pst_weight, 0.9);
        assert_eq!(weights.king_safety_weight, 1.1);
    }
}
