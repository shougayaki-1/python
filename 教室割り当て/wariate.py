# ===============================================================
# 学校祭 教室割り当て自動化プログラム v3.4 (Python Script版)
# ===============================================================

# 必要なライブラリをインポート
import pandas as pd
import numpy as np
import copy
import datetime
import os # ファイル存在確認のため
import traceback # エラー追跡のため

# ---------------------------------------------------------------
# 設定 (CONFIG)
# ---------------------------------------------------------------
CONFIG = {
  'MAX_PATTERNS': 3,                 # 生成する最大パターン数
  'BRANCHING_FACTOR': 2,             # 詳細探索時の分岐数
  'CATEGORY_DIVERSITY_WEIGHT': 0.5,  # 企画カテゴリ分散を評価する重み (0に近いほど机不足を優先)
  'GRADE_DIVERSITY_WEIGHT': 0.5,     # 学年分散を評価する重み (0に近いほど机不足を優先)
  'EXPANSION_PAIRS': {
    # 展開企画で使用できる教室のペア (キーは文字列の '1', '2', '3')
    '1': [['国語2', '国語3'], ['社会4', '社会5']],
    '2': [['数学3', '数学4'], ['数学8', '数学9'], ['理科1', '理科2']],
    '3': [['英語5', '英語6']],
  },
}

# ---------------------------------------------------------------
# ログ出力関数
# ---------------------------------------------------------------
def log(level, message):
  """コンソールにログを出力する"""
  timestamp = datetime.datetime.now().strftime('%H:%M:%S')
  print(f"[{timestamp}][{level}] {message}")

# ---------------------------------------------------------------
# グローバル変数として all_projects と all_classrooms を定義
# ---------------------------------------------------------------
all_projects = []
all_classrooms = []

# ---------------------------------------------------------------
# ファイル読み込み関数
# ---------------------------------------------------------------
def load_data_files():
    global all_projects, all_classrooms
    print("「企画リスト.csv」と「教室リスト.csv」をこのスクリプトと同じディレクトリに配置してください。")

    project_list_file = '企画リスト.csv'
    classroom_list_file = '教室リスト.csv'

    if not os.path.exists(project_list_file) or not os.path.exists(classroom_list_file):
        log('ERROR', f"「{project_list_file}」または「{classroom_list_file}」が見つかりません。処理を中断します。")
        raise FileNotFoundError(f"エラー: 「{project_list_file}」または「{classroom_list_file}」が見つかりません。")

    try:
        projects_df = pd.read_csv(project_list_file)
        classrooms_df = pd.read_csv(classroom_list_file)
        
        all_projects = projects_df.to_dict('records')
        all_classrooms = classrooms_df.to_dict('records')

        log('INFO', f"データ読み込み完了: {len(all_projects)}企画, {len(all_classrooms)}教室")
    except Exception as e:
        log('ERROR', f"CSVファイルの読み込みに失敗しました: {e}")
        traceback.print_exc()
        raise # エラーを再送出してプログラムを停止

# ===============================================================
# アルゴリズムとヘルパー関数
# (このセクションのコードは変更ありません)
# ===============================================================

def standard_deviation(arr):
  """リストの標準偏差を計算する"""
  if len(arr) < 2:
    return 0
  return np.std(arr)

def calculate_supplies(classrooms):
  """階ごとの既存備品数を計算する"""
  supplies = {1: {'d': 0, 'c': 0}, 2: {'d': 0, 'c': 0}, 3: {'d': 0, 'c': 0}}
  for c in classrooms:
      floor = int(c['階'])
      supplies[floor]['d'] += int(c['既存机数'])
      supplies[floor]['c'] += int(c['既存椅子数'])
  return supplies

def get_project_priority(project):
  """企画の優先順位を決定する"""
  p_type = project['企画種別']
  priorities = {
    '本部': 1,
    '展開企画': 2,
    '中庭控室': 3,
  }
  if p_type in ['フォトスポット', '休憩所']:
    return 100
  return priorities.get(p_type, 99)


def calculate_final_score(state, supplies, current_all_projects, current_all_classrooms):
    demands = {1: {'d': 0, 'c': 0, 'count': 0}, 2: {'d': 0, 'c': 0, 'count': 0}, 3: {'d': 0, 'c': 0, 'count': 0}}
    category_distribution = {1: {}, 2: {}, 3: {}}
    grade_distribution = {1: {}, 2: {}, 3: {}}

    project_map = {p['企画名']: p for p in current_all_projects}
    classroom_map = {c['教室名']: c for c in current_all_classrooms}

    for a in state['assignments']:
        p = project_map.get(a['project'])
        c = classroom_map.get(a['classrooms'][0])
        if p and c:
            floor = int(c['階'])
            demands[floor]['d'] += int(p['必要机数'])
            demands[floor]['c'] += int(p['必要椅子数'])
            demands[floor]['count'] += 1

            category = p.get('企画概要')
            if category:
                category_distribution[floor].setdefault(category, 0)
                category_distribution[floor][category] += 1
            
            grade = p.get('担当学年')
            if grade:
                grade_distribution[floor].setdefault(grade, 0)
                grade_distribution[floor][grade] += 1

    desk_penalty = 0
    desk_shortages_raw = []
    has_shortage = False
    for f in [1, 2, 3]:
        shortage = demands[f]['d'] - supplies[f]['d']
        if shortage > 0:
            desk_penalty += 10000 * shortage
            has_shortage = True
        desk_shortages_raw.append(shortage)

    if has_shortage:
        return desk_penalty

    surpluses = [-s for s in desk_shortages_raw]
    desk_penalty += standard_deviation(surpluses)

    category_penalty = 0
    all_categories_in_data = set(p['企画概要'] for p in current_all_projects if pd.notna(p.get('企画概要')))
    for category in all_categories_in_data:
        counts_per_floor = [category_distribution[f].get(category, 0) for f in [1, 2, 3]]
        category_penalty += standard_deviation(counts_per_floor)

    grade_penalty = 0
    all_grades_in_data = set(p['担当学年'] for p in current_all_projects if pd.notna(p.get('担当学年')))
    for grade in all_grades_in_data:
        counts_per_floor = [grade_distribution[f].get(grade, 0) for f in [1, 2, 3]]
        grade_penalty += standard_deviation(counts_per_floor)

    final_score = (desk_penalty + 
                   (category_penalty * CONFIG['CATEGORY_DIVERSITY_WEIGHT']) +
                   (grade_penalty * CONFIG['GRADE_DIVERSITY_WEIGHT']))
    return final_score


def is_placement_valid(project, classrooms_to_assign, state, current_all_classrooms):
    is_noisy = project.get('騒音予想') == True
    causes_lines = project.get('行列予想') == True

    for classroom_in_set in classrooms_to_assign: # 変数名変更 (classroom -> classroom_in_set)
        if classroom_in_set['isAssigned']:
            return False 
        is_courtyard_adjacent = classroom_in_set.get('中庭隣接(TRUE/FALSE)') == True
        is_corner = classroom_in_set.get('角部屋(TRUE/FALSE)') == True
        if is_noisy and is_courtyard_adjacent:
            return False 
        if causes_lines and is_corner:
            return False 

    project_name = project['企画名']
    if project_name in ['フォトスポット', '休憩所']:
        other_project_name = '休憩所' if project_name == 'フォトスポット' else 'フォトスポット'
        other_project_assignment = next((a for a in state['assignments'] if a['project'] == other_project_name), None)
        
        if other_project_assignment:
            classroom_map = {c['教室名']: c for c in current_all_classrooms}
            other_project_classroom = classroom_map.get(other_project_assignment['classrooms'][0])
            
            target_floor = int(classrooms_to_assign[0]['階'])
            if other_project_classroom and int(other_project_classroom['階']) == target_floor:
                return False 
    return True


def apply_placement(state, project, classrooms_to_assign):
    new_state = copy.deepcopy(state)
    assigned_classroom_names = [c['教室名'] for c in classrooms_to_assign]
    new_state['assignments'].append({'project': project['企画名'], 'classrooms': assigned_classroom_names})
    
    for c_name in assigned_classroom_names:
        for classroom_state in new_state['classrooms']:
            if classroom_state['教室名'] == c_name:
                classroom_state['isAssigned'] = True
                break
    
    if project['企画種別'] == '展開企画':
        floor_key = str(classrooms_to_assign[0]['階'])
        pair_to_remove = sorted(assigned_classroom_names)
        if floor_key in new_state['available_pairs']:
            new_state['available_pairs'][floor_key] = [
                p for p in new_state['available_pairs'][floor_key] if sorted(p) != pair_to_remove
            ]
    return new_state


def get_placement_options(project, state, supplies, current_all_projects, current_all_classrooms):
    options = []
    project_type = project['企画種別']
    potential_classroom_sets = []

    if project_type == '本部':
        classroom = next((c for c in state['classrooms'] if c['教室名'] == '数学1' and not c['isAssigned']), None)
        if classroom:
            potential_classroom_sets.append([classroom])
    elif project_type == '中庭控室':
        for c in state['classrooms']:
            if int(c['階']) == 1 and not c['isAssigned']:
                potential_classroom_sets.append([c])
    elif project_type == '展開企画':
        for floor_key, pairs in state['available_pairs'].items():
            for pair in pairs:
                c1 = next((c for c in state['classrooms'] if c['教室名'] == pair[0]), None)
                c2 = next((c for c in state['classrooms'] if c['教室名'] == pair[1]), None)
                if c1 and c2 and not c1['isAssigned'] and not c2['isAssigned']:
                    if str(c1['階']) == floor_key and str(c2['階']) == floor_key :
                         potential_classroom_sets.append([c1, c2])
    else: 
        for c in state['classrooms']:
            if not c['isAssigned']:
                potential_classroom_sets.append([c])
    
    for classroom_set in potential_classroom_sets:
        if is_placement_valid(project, classroom_set, state, current_all_classrooms):
            future_state = apply_placement(state, project, classroom_set)
            score = calculate_final_score(future_state, supplies, current_all_projects, current_all_classrooms)
            options.append({'classrooms': classroom_set, 'score': score})
    return options

# ------------------------------------------------------------------------------------------
# 詳細探索アルゴリズム本体
# ------------------------------------------------------------------------------------------
def assign_classrooms_recursive(initial_classrooms_data, initial_projects_data): # 引数名変更
    final_patterns = []
    
    sorted_projects = sorted(initial_projects_data, key=get_project_priority)
    supplies = calculate_supplies(initial_classrooms_data)
    
    # initial_classrooms_data と initial_projects_data をクロージャ内で使用
    def find_assignments_recursive(current_project_index, current_state):
        if current_project_index >= len(sorted_projects):
            score = calculate_final_score(current_state, supplies, initial_projects_data, initial_classrooms_data)
            log('DEBUG', f"パターン発見。スコア: {score:.4f}")
            if score < 10000:
                final_patterns.append({'state': current_state, 'score': score})
                final_patterns.sort(key=lambda x: x['score'])
                if len(final_patterns) > CONFIG['MAX_PATTERNS']:
                    final_patterns.pop()
            return

        project = sorted_projects[current_project_index]
        log('DEBUG', f"[{current_project_index + 1}/{len(sorted_projects)}] 企画「{project['企画名']}」の配置を探索中...")
        
        placement_options = get_placement_options(project, current_state, supplies, initial_projects_data, initial_classrooms_data)

        if not placement_options:
            log('WARN', f" -> 企画「{project['企画名']}」を配置できる場所がありません。")
            next_state_unassigned = copy.deepcopy(current_state)
            unassigned_project_info = {key: project[key] for key in project if key != 'isAssigned'}
            next_state_unassigned['unassigned_projects'].append(unassigned_project_info)
            find_assignments_recursive(current_project_index + 1, next_state_unassigned)
            return

        placement_options.sort(key=lambda x: x['score'])
        branches_to_explore = placement_options[:CONFIG['BRANCHING_FACTOR']]
        log('DEBUG', f" -> 候補数: {len(placement_options)}件。探索する分岐数: {len(branches_to_explore)}件")
        
        for option in branches_to_explore:
            class_names = ', '.join([c['教室名'] for c in option['classrooms']])
            log('DEBUG', f"  -> 分岐: 教室「{class_names}」に配置して再帰 (スコア: {option['score']:.4f})")
            next_state = apply_placement(current_state, project, option['classrooms'])
            find_assignments_recursive(current_project_index + 1, next_state)

    initial_state = {
        'classrooms': [dict(c, isAssigned=False) for c in initial_classrooms_data],
        'assignments': [],
        'available_pairs': copy.deepcopy(CONFIG['EXPANSION_PAIRS']),
        'unassigned_projects': [],
    }
    
    find_assignments_recursive(0, initial_state)
    return final_patterns


# ------------------------------------------------------------------------------------------
# 結果出力関数
# ------------------------------------------------------------------------------------------
def write_results_to_csv(pattern, current_all_classrooms, current_all_projects, pattern_index):
    assignments = pattern['state']['assignments']
    unassigned_projects_list = pattern['state']['unassigned_projects']
    score = pattern['score']
    
    output_data = []
    project_map = {p['企画名']: p for p in current_all_projects}
    classroom_map = {c['教室名']: c for c in current_all_classrooms}
    
    floor_demands = {1: {'d': 0, 'c': 0}, 2: {'d': 0, 'c': 0}, 3: {'d': 0, 'c': 0}}
    
    for a in assignments:
        project = project_map.get(a['project'])
        num_classrooms = len(a['classrooms'])
        for c_name in a['classrooms']:
            classroom_obj = classroom_map.get(c_name) # 変数名変更 (classroom -> classroom_obj)
            if project and classroom_obj:
                floor = int(classroom_obj['階'])
                desks = int(project['必要机数']) / num_classrooms
                chairs = int(project['必要椅子数']) / num_classrooms
                output_data.append([
                    floor, c_name, project['企画名'], project.get('担当学年', ''),
                    project.get('企画概要', ''), desks, chairs,
                ])
                floor_demands[floor]['d'] += desks
                floor_demands[floor]['c'] += chairs

    headers = ['階', '教室名', '企画名', '担当学年', '企画概要', '必要机数', '必要椅子数']
    result_df = pd.DataFrame(output_data, columns=headers)
    result_df = result_df.sort_values(by=['階', '教室名'])

    summary_data = []
    floor_supplies = calculate_supplies(current_all_classrooms)
    final_demands = {1: {'count': 0}, 2: {'count': 0}, 3: {'count': 0}}
    for a in assignments:
        first_classroom_name = a['classrooms'][0]
        c = classroom_map.get(first_classroom_name)
        if c:
            final_demands[int(c['階'])]['count'] += 1

    for f in [1, 2, 3]:
        sup_d, dem_d_float = floor_supplies[f]['d'], floor_demands[f]['d']
        dem_d = round(dem_d_float)
        diff_d = dem_d - sup_d
        rate_d = (diff_d / sup_d) if sup_d > 0 and diff_d > 0 else 0
        project_count = final_demands[f]['count']
        shortage_per_project = (diff_d / project_count) if project_count > 0 and diff_d > 0 else 0
        
        sup_c, dem_c_float = floor_supplies[f]['c'], floor_demands[f]['c']
        dem_c = round(dem_c_float)
        diff_c = dem_c - sup_c
        rate_c = (diff_c / sup_c) if sup_c > 0 and diff_c > 0 else 0
        
        summary_data.append([
            f"{f}階", sup_d, dem_d, max(0, diff_d), f"{rate_d:.1%}", f"{shortage_per_project:.1f}",
            sup_c, dem_c, max(0, diff_c), f"{rate_c:.1%}"
        ])

    summary_headers = ['階', '既存机数', '必要机数', '机不足数', '机不足率', '一人あたり机不足数', '既存椅子数', '必要椅子数', '椅子不足数', '椅子不足率']
    summary_df = pd.DataFrame(summary_data, columns=summary_headers)

    if unassigned_projects_list:
        unassigned_df_data = [{'未割り当ての企画': p.get('企画名', '不明な企画')} for p in unassigned_projects_list]
        unassigned_df = pd.DataFrame(unassigned_df_data)
    else:
        unassigned_df = pd.DataFrame(columns=['未割り当ての企画'])

    filename = f'結果パターン{pattern_index}.csv'
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f_out:
        f_out.write(f"割り当てスコア (均等化): {score:.4f} (低いほど良い)\n\n")
        f_out.write("■ 割り当て結果\n")
        result_df.to_csv(f_out, index=False)
        f_out.write("\n■ 備品過不足サマリー\n")
        summary_df.to_csv(f_out, index=False)
        f_out.write("\n")
        if not unassigned_df.empty:
            f_out.write("■ 未割り当ての企画\n")
            unassigned_df.to_csv(f_out, index=False)
    log('INFO', f"結果を {filename} に出力しました。")
    return filename

# ---------------------------------------------------------------
# メイン実行関数
# ---------------------------------------------------------------
def run_assignment_process(classrooms_data, projects_data): # 引数名変更
    try:
        start_time = datetime.datetime.now()
        log('INFO', "処理開始 (モード: 詳細探索)")
        
        log('INFO', '詳細探索アルゴリズムの実行開始')
        # グローバル変数ではなく、引数で渡されたデータを使用
        assignment_patterns = assign_classrooms_recursive(classrooms_data, projects_data)

        if not assignment_patterns:
            log('WARN', '有効な割り当てパターンが見つかりませんでした。')
            print("\n【処理終了】有効な割り当てパターンを一つも見つけることができませんでした。制約が厳しすぎるか、CONFIG設定を見直してください。")
            return

        log('INFO', f"アルゴリズム完了。{len(assignment_patterns)}個のパターンを発見。")
        log('INFO', '結果の出力開始')
        
        output_files = []
        for i, pattern in enumerate(assignment_patterns):
            log('INFO', f"パターン{i + 1}をCSVに出力中... (スコア: {pattern['score']:.4f})")
            # グローバル変数ではなく、引数で渡されたデータを使用
            filename = write_results_to_csv(pattern, classrooms_data, projects_data, i + 1)
            output_files.append(filename)

        log('INFO', '結果の出力完了')
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        final_message = f"処理完了 ({len(assignment_patterns)}パターン生成, {duration:.1f}秒)。"
        log('INFO', final_message)
        print(f"\n【処理完了】結果ファイルが生成されました: {', '.join(output_files)}")
        print("これらのファイルは、このスクリプトと同じディレクトリに保存されています。")

    except Exception as e:
        log('ERROR', f"割り当て処理中に致命的なエラーが発生しました: {e}")
        traceback.print_exc()
        print(f"処理中にエラーが発生しました。詳細はログを確認してください: {e}")


# ===============================================================
# スクリプト実行
# ===============================================================
if __name__ == "__main__":
    try:
        # 1. ファイルデータの読み込み
        load_data_files() # all_projects と all_classrooms がグローバルに設定される

        # 2. 割り当て処理の実行
        #    グローバル変数 all_projects と all_classrooms を引数として渡す
        if all_projects and all_classrooms: # データが正常に読み込まれた場合のみ実行
             run_assignment_process(all_classrooms, all_projects)
        else:
            log('CRITICAL', "データが読み込まれていないため、割り当て処理を実行できませんでした。")
            print("エラー: データファイルが正しく読み込まれませんでした。上記のログを確認してください。")

    except FileNotFoundError:
        # load_data_files内でログ出力済みなので、ここでは簡潔に
        print("必要なCSVファイルが見つからなかったため、処理を終了します。")
    except Exception as e:
        log('CRITICAL', f"スクリプトの実行中に予期せぬ致命的なエラーが発生しました: {e}")
        traceback.print_exc()
        print(f"処理中に予期せぬ致命的なエラーが発生しました。詳細はログを確認してください: {e}")