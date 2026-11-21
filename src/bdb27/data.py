from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = {
    'players': 'nfl_id display_name draft_year nfl_position birth_date college_name college_conference draft_round draft_pick_within_round draft_overall_pick',
    'combine_results': 'draft_year nfl_id combine_position combine_height combine_weight hand_size arm_length wing_span ten_yd_split forty vertical broad_jump three_cone short_shuttle bench_reps ngs_athleticism_score ngs_college_production_score ngs_final_score',
    'combine_tracking': 'draft_year event_id nfl_id entity_type time drill_type drill_name attempt x y s a dis dir',
    'player_career_successes': 'nfl_id career_offensive_snaps career_defensive_snaps career_special_teams_snaps career_games_active career_games_started ap_all_pro_1st_team ap_all_pro_2nd_team pro_bowl_original_ballot',
    'games': 'game_id game_key season season_type week home_team_abbr visitor_team_abbr',
    'player_play': 'game_id play_id team_abbr nfl_id lined_up_position play_description quarter down yards_to_go possession_team yardline_side yardline_number game_clock pre_snap_home_score pre_snap_visitor_score play_direction pre_snap_home_team_win_probability pre_snap_visitor_team_win_probability home_team_win_probability_added visitor_team_win_probility_added expected_points_added expected_points offense_formation receiver_alignment pass_result pass_length play_nullified_by_penalty penalty_yards pre_penalty_yards_gained yards_gained team_coverage_man_zone team_coverage_type in_motion_at_ball_snap target rec_yards rush_yards yards_after_catch fumble fumble_lost route_ran separation_at_pass_forward cushion expected_yards_after_catch dropback_duration extended_sack_allowed pass_rushers_encountered peak_pressure_probability_allowed pressure_allowed sack_allowed time_to_pressure_allowed blitzing player_get_off sack tackle assist caused_forced_fumble recovered_fumble tackle_for_loss time_to_pressure time_to_qb_hurry coverage_assignment coverage_assignment_at_snap quick_pressure unblocked_pressure',
    'game_tracking': 'game_id play_id nfl_id time x y s a dis o dir event',
}

def load_table(root, name, **kwargs):
    return pd.read_csv(Path(root) / (name + '.csv'), **kwargs)

def regular_plays(root):
    games = load_table(root, 'games')
    plays = load_table(root, 'player_play')
    return plays.merge(games.loc[games.season_type.eq('REG'), ['game_id', 'season']], on='game_id').query("play_nullified_by_penalty != 'Y'")
