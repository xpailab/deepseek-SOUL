
  // 安全的 markdown 渲染（CDN 失败时降级为纯文本转义）
  function renderMD(text) {
    if (typeof marked === 'undefined' || window._markedFail) {
      var d = document.createElement('div'); d.textContent = text; return d.innerHTML;
    }
    if (!window._markedInit) {
      try{
        if(typeof marked.setOptions==='function') marked.setOptions({gfm:true,breaks:false});
        else if(typeof marked.use==='function') marked.use({gfm:true,breaks:false});
      }catch(e){}
      window._markedInit = true;
    }
    // 仅保护 __dunder__ 变量名不被 marked 吃掉——marked 不会将词内下划线 (file_name, deepseek-SOUL) 当作斜体
    var clean = text.replace(/__([a-zA-Z0-9_]+)__/g, '<code>$1</code>');
    if(typeof marked.parse==='function') return marked.parse(clean);
    else return marked(clean);
  }

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  let ws = null, sessionId = '', wsReady = false, reconnectTimer = null;
  let agentCards = {}, agentsRow = null;

  
  function escHtml(s) { var d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function toast(t) {
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;bottom:80px;left:50%;transform:translateX(-50%);background:#ef4444;color:#fff;padding:8px 18px;border-radius:8px;font-size:.82rem;z-index:999';
    el.textContent = t; document.body.appendChild(el);
    setTimeout(()=>{el.style.opacity='0';el.style.transition='opacity .5s'},2000);
    setTimeout(()=>el.remove(),2500);
  }

  // ========== 会话管理 ==========
  let sessions = [];      // {id, title, status, messages:[], serverSid:''}
  let activeSid = null;

  function loadSessions() { try{ sessions=JSON.parse(localStorage.getItem('ds_sessions')||'[]'); }catch(e){ sessions=[]; } }
  function saveSessions() { localStorage.setItem('ds_sessions', JSON.stringify(sessions)); }
  function getSession(id) { return sessions.find(function(s){ return s.id===id; }); }

  function renderSidebar() {
    var list=document.getElementById('sessionList');
    list.innerHTML = sessions.map(function(s){
      var cls=(s.id===activeSid)?' active':'';
      var sc=s.status==='running'?'running':(s.status==='error'?'error':'done');
      var sl=s.status==='running'?'执行中':(s.status==='error'?'出错':'完成');
      return '<div class="session-item'+cls+'" data-sid="'+s.id+'" onclick="switchSession(\x27'+s.id+'\x27)">'+
        '<span class="s-title">'+(s.title||'新对话')+'</span>'+
        '<span class="s-status '+sc+'">'+sl+'</span>'+
        '<span class="s-delete" onclick="event.stopPropagation();deleteSession(\x27'+s.id+'\x27)">x</span></div>';
    }).join('');
    document.getElementById('msgCount').textContent=sessions.length+' 个会话';
  }

  function switchSession(id) {
    saveCurrentMessages();
    activeSid=id;
    var s=getSession(id); if(!s) return;
    sessionId=s.serverSid||'';
    document.getElementById('welcome')?.remove();
    document.getElementById('msgList').innerHTML=s.messages.map(function(m){return m.html;}).join('');
    document.getElementById('sendBtn').disabled=(s.status==='running');
    simpleMsg=null; agentCards={}; agentsRow=null;
    renderSidebar(); scrollDown();
  }

  function saveCurrentMessages() {
    if(!activeSid) return;
    var s=getSession(activeSid); if(!s) return;
    var msgs=[];
    document.querySelectorAll('#msgList .msg').forEach(function(el){ msgs.push({html:el.outerHTML}); });
    if(msgs.length>50) msgs=msgs.slice(-50);
    s.messages=msgs; saveSessions();
  }

  function deleteSession(id) {
    if(!confirm('删除此会话？')) return;
    sessions=sessions.filter(function(s){return s.id!==id;});
    saveSessions();
    if(activeSid===id){ activeSid=null; document.getElementById('msgList').innerHTML='<div class="welcome" id="welcome"><div class="welcome-icon">&#9670;</div><h2>有什么我可以帮你的？</h2></div>'; }
    renderSidebar();
  }

  function updateSessionStatus(id,status) {
    var s=getSession(id); if(s){ s.status=status; saveSessions(); renderSidebar(); }
  }

  async function init() {
    loadSessions(); renderSidebar();
    if(sessions.length>0){ var last=sessions[sessions.length-1]; if(last.status!=='running') switchSession(last.id); else activeSid=last.id; }
    connect();
    try{ var r=await fetch('/api/status'); var d=await r.json(); if(d.llm) document.getElementById('modelName').textContent=d.llm.model||'-'; }catch(e){}
  }
  function connect() {
    if(ws && (ws.readyState===WebSocket.CONNECTING || ws.readyState===WebSocket.OPEN)) return;
    ws = new WebSocket(`${protocol}//${location.host}/ws/chat`); ws.sessionId = sessionId;
    ws.onopen = () => { wsReady = true; document.getElementById('sendBtn').disabled = false; document.getElementById('input').focus(); };
    ws.onmessage = handleMessage;
    ws.onerror = () => { wsReady = false; };
    ws.onclose = () => { wsReady = false; if(reconnectTimer) clearTimeout(reconnectTimer); reconnectTimer = setTimeout(connect,2000); };
  }

  let simpleMsg = null;  // 简单对话的当前消息元素

  function handleMessage(e) {
    let d; try{ d = JSON.parse(e.data); }catch(_){ return; }
    const sid = d.stream_id || '';

    // === 简单对话模式 (无 stream_id，旧格式) ===
    if(!sid){
      if(d.c){
        if(!simpleMsg){
          document.getElementById('welcome')?.remove();
          simpleMsg = document.createElement('div'); simpleMsg.className = 'msg assistant';
          simpleMsg.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
          document.getElementById('msgList').appendChild(simpleMsg);
        }
        var cur = simpleMsg.querySelector('.msg-text').getAttribute('data-raw') || ''; cur += d.c; simpleMsg.querySelector('.msg-text').setAttribute('data-raw', cur); simpleMsg.querySelector('.msg-text').innerHTML = renderMD(cur);
        scrollDown();
      }
      if(d.t){
        if(!simpleMsg){
          document.getElementById('welcome')?.remove();
          simpleMsg = document.createElement('div'); simpleMsg.className = 'msg assistant';
          simpleMsg.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
          document.getElementById('msgList').appendChild(simpleMsg);
        }
        var tl = document.createElement('div');
        tl.style.cssText = 'font-size:.75rem;margin:3px 0;padding:5px 10px;background:#f3f4f6;border-radius:5px;border-left:3px solid var(--accent)';
        var argsStr = d.args ? (typeof d.args==='string'?d.args.slice(0,150):JSON.stringify(d.args||{}).slice(0,150)) : '';
        tl.innerHTML = '<span style=\"color:var(--accent);font-weight:600\">&#9881; ' + (d.t||'tool') + '</span>' + (argsStr?' <span style=\"color:var(--text-dim);font-size:.68rem;font-family:monospace\">' + escHtml(argsStr) + '</span>':'');
        simpleMsg.querySelector('.msg-body').appendChild(tl);
        scrollDown();
      }
      if(d.r){
        var rl = document.createElement('div');
        var ok = d.r.ok;
        var txt = (d.r.text||'').slice(0,400);
        if((d.r.text||'').length > 400) txt += '...';
        rl.style.cssText = 'font-size:.72rem;margin:1px 0 3px 8px;padding:4px 10px;border-radius:4px;color:' + (ok?'var(--success)':'var(--danger)') + ';background:' + (ok?'#f0fdf4':'#fef2f2') + ';white-space:pre-wrap;word-break:break-all';
        rl.textContent = (ok?'✓ ':'✗ ') + txt;
        if(simpleMsg) simpleMsg.querySelector('.msg-body').appendChild(rl);
        scrollDown();
      }
      if(d.f){
        simpleMsg = null;
        document.getElementById('sendBtn').disabled = false;
        document.getElementById('input').focus();
        document.getElementById('msgCount').textContent = document.querySelectorAll('.msg').length + ' 条消息';
        // 标记会话完成
        if(activeSid) updateSessionStatus(activeSid, d.f==='error'?'error':'done');
        saveCurrentMessages();
      }
      return;
    }

    // === 并行模式 (有 stream_id) ===

    // Meta: 并行启动
    if(sid === '_meta' && d.type === 'start'){
      document.getElementById('welcome')?.remove();
      const count = d.count || 1;
      // 单 Agent → 不显示卡片，用正常消息流
      if(count <= 1){
        const div = document.createElement('div'); div.className = 'msg assistant';
        div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\"></div></div>';
        div.id = 'singleAgentMsg';
        document.getElementById('msgList').appendChild(div);
        agentCards['0'] = {msgDiv: div, body: div.querySelector('.msg-text')};
        scrollDown(); return;
      }
      // 多 Agent → 横向卡片布局（默认折叠）
      agentsRow = document.createElement('div'); agentsRow.className = 'agents-row';
      document.getElementById('msgList').appendChild(agentsRow);
      (d.approaches||[]).forEach((name,i) => {
        const card = document.createElement('div'); card.className = 'agent-card';
        card.innerHTML = '<div class=\"card-head\"><span class=\"arrow\">&#9654;</span><span class=\"c-name\">' + name + '</span><span class=\"c-status\">等待</span></div><div class=\"card-body collapsed\"></div>';
        card.querySelector('.card-head').onclick = function(){
          const body = card.querySelector('.card-body');
          const arrow = card.querySelector('.arrow');
          body.classList.toggle('collapsed');
          arrow.classList.toggle('open');
        };
        agentsRow.appendChild(card);
        agentCards[i] = card;
      });
      scrollDown(); return;
    }

    // Meta: 完成
    if(sid === '_meta' && (d.type === 'finished' || d.type === 'partial')){
      const w = d.winner !== undefined ? String(d.winner) : null;
      const hasCards = agentsRow !== null;

      if(hasCards){
        Object.keys(agentCards).forEach(k => {
          const card = agentCards[k];
          const st = card.querySelector('.c-status');
          if(k === w){ card.classList.add('winner'); st.textContent = '采用'; st.style.color = '#065f46'; }
          else { card.classList.add('loser'); st.textContent = '未用'; st.style.color = '#991b1b'; }
        });
        // 追加最终结果（含任务报告）
        const div = document.createElement('div'); div.className = 'msg assistant';
        const winnerLabel = d.winner_name ? '【采用: ' + d.winner_name + '】' : '';
        div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + winnerLabel + (d.content ? renderMD(d.content) : '完成') + '</div></div>';
        document.getElementById('msgList').appendChild(div);
      }
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('input').focus();
      document.getElementById('msgCount').textContent = document.querySelectorAll('.msg').length + ' 条消息';
      agentCards = {}; agentsRow = null;
      if(activeSid) updateSessionStatus(activeSid, 'done');
      saveCurrentMessages();
      scrollDown(); return;
    }

    // Meta: 阶段计划
    if(sid === '_meta' && d.type === 'stage_plan'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-plan';
      let stagesHtml = '<div style=\"background:var(--accent-light);border:1px solid var(--accent);border-radius:8px;padding:12px;margin:8px 0;\">';
      stagesHtml += '<div style=\"font-weight:600;color:var(--accent);margin-bottom:8px;\">📋 任务执行计划</div>';
      stagesHtml += '<div style=\"font-size:.85rem;color:var(--text-secondary);margin-bottom:8px;\">' + (d.message||'') + '</div>';
      if(d.plan && d.plan.stages){
        stagesHtml += '<div style=\"display:flex;flex-direction:column;gap:4px;\">';
        d.plan.stages.forEach((stage, idx) => {
          stagesHtml += '<div style=\"display:flex;align-items:center;gap:8px;padding:6px 8px;background:var(--bg);border-radius:4px;\">';
          stagesHtml += '<span style=\"background:var(--accent);color:white;padding:2px 6px;border-radius:3px;font-size:.7rem;\">阶段' + (idx+1) + '</span>';
          stagesHtml += '<span style=\"font-weight:500;\">' + stage.name + '</span>';
          stagesHtml += '<span style=\"margin-left:auto;font-size:.75rem;color:var(--text-dim);\">预计' + stage.estimated_tools + '步</span>';
          stagesHtml += '</div>';
        });
        stagesHtml += '</div>';
      }
      stagesHtml += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + stagesHtml + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 阶段开始
    if(sid === '_meta' && d.type === 'stage_start'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-start';
      let html = '<div style=\"background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:10px 12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">▶️</span>';
      html += '<span style=\"font-weight:600;color:#166534;\">开始执行: ' + (d.stage?.name||'当前阶段') + '</span>';
      html += '<span style=\"margin-left:auto;font-size:.75rem;color:#22c55e;\">' + (d.progress||'') + '</span>';
      html += '</div>';
      if(d.stage?.description){
        html += '<div style=\"margin-top:6px;font-size:.85rem;color:var(--text-secondary);padding-left:28px;\">' + d.stage.description + '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 阶段完成
    if(sid === '_meta' && d.type === 'stage_complete'){
      const div = document.createElement('div'); div.className = 'msg assistant stage-complete';
      let html = '<div style=\"background:#f0f9ff;border:1px solid #7dd3fc;border-radius:8px;padding:10px 12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">✅</span>';
      html += '<span style=\"font-weight:600;color:#0369a1;\">阶段完成: ' + (d.stage?.name||'') + '</span>';
      html += '</div>';
      if(d.summary){
        html += '<div style=\"margin-top:6px;font-size:.85rem;color:var(--text);padding-left:28px;\">' + d.summary + '</div>';
      }
      if(d.artifacts && d.artifacts.length > 0){
        html += '<div style=\"margin-top:6px;font-size:.8rem;color:var(--text-dim);padding-left:28px;\">';
        html += '📁 交付物: ' + d.artifacts.join(', ');
        html += '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      scrollDown(); return;
    }

    // Meta: 等待确认
    if(sid === '_meta' && d.type === 'await_confirm'){
      const div = document.createElement('div'); div.className = 'msg assistant await-confirm';
      let html = '<div style=\"background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;padding:12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:8px;\">';
      html += '<span style=\"font-size:1.2rem;\">⏸️</span>';
      html += '<span style=\"font-weight:600;color:#92400e;\">' + (d.message||'阶段完成') + '</span>';
      html += '</div>';
      if(d.next_stage){
        html += '<div style=\"font-size:.85rem;color:var(--text-secondary);padding-left:28px;margin-bottom:8px;\">';
        html += '下一阶段: <strong>' + d.next_stage.name + '</strong> - ' + d.next_stage.description;
        html += '</div>';
      }
      html += '<div style=\"padding-left:28px;\">';
      html += '<button onclick=\"continueStage()\" style=\"background:#f59e0b;color:white;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:.85rem;\">继续下一阶段 →</button>';
      html += '<span style=\"margin-left:12px;font-size:.8rem;color:var(--text-dim);\">' + (d.progress||'') + '</span>';
      html += '</div>';
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      div.id = 'awaitConfirmMsg';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = true;
      scrollDown(); return;
    }

    // Meta: 所有阶段完成
    if(sid === '_meta' && d.type === 'all_stages_complete'){
      const div = document.createElement('div'); div.className = 'msg assistant all-complete';
      let html = '<div style=\"background:#d1fae5;border:1px solid #34d399;border-radius:8px;padding:12px;margin:8px 0;\">';
      html += '<div style=\"display:flex;align-items:center;gap:8px;\">';
      html += '<span style=\"font-size:1.5rem;\">🎉</span>';
      html += '<span style=\"font-weight:600;color:#065f46;font-size:1.1rem;\">所有阶段已完成！</span>';
      html += '</div>';
      if(d.plan && d.plan.stages){
        html += '<div style=\"margin-top:8px;font-size:.85rem;color:var(--text);padding-left:36px;\">';
        html += '共完成 ' + d.plan.stages.length + ' 个阶段';
        html += '</div>';
      }
      html += '</div>';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\">' + html + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = false;
      scrollDown(); return;
    }

    // Meta: 错误
    if(sid === '_meta' && d.type === 'error'){
      const div = document.createElement('div'); div.className = 'msg assistant';
      div.innerHTML = '<div class=\"msg-body\"><div class=\"msg-text\" style=\"color:var(--danger)\">' + (d.content ? renderMD(d.content) : '错误') + '</div></div>';
      document.getElementById('msgList').appendChild(div);
      document.getElementById('sendBtn').disabled = false;
      agentCards = {}; agentsRow = null;
      scrollDown(); return;
    }

    // Agent 事件
    if(sid && sid !== '_meta' && agentCards[sid]){
      const entry = agentCards[sid];
      // 单 Agent 消息模式
      if(entry.msgDiv){
        if(d.type === 'content'){ entry.body.innerHTML = renderMD((entry.body.textContent||'') + d.content); }
        else if(d.type === 'tool'){
          const tl = document.createElement('div');
          tl.style.cssText = 'font-size:.7rem;color:var(--text-dim);margin:2px 0';
          tl.textContent = '...' + d.tool; entry.body.appendChild(tl);
        }
        else if(d.type === 'result'){
          const rl = document.createElement('div');
          rl.style.cssText = 'font-size:.68rem;margin:1px 0;color:' + (d.success?'var(--success)':'var(--danger)');
          rl.textContent = (d.success?'OK ':'FAIL ') + (d.tool||'') + ': ' + (d.text||'').slice(0,60);
          entry.body.appendChild(rl);
        }
      }
      // 多 Agent 卡片模式
      else {
        const body = entry.querySelector('.card-body');
        const status = entry.querySelector('.c-status');
        if(d.type === 'agent_start'){ status.textContent = '执行'; status.style.color = '#3b82f6'; }
        else if(d.type === 'content'){
          if(body.classList.contains('collapsed')){
            body.classList.remove('collapsed');
            entry.querySelector('.arrow').classList.add('open');
          }
          // 追加到同一个文本行，不创建新 div
          let last = body.querySelector('.c-text:last-child');
          if(!last){ last = document.createElement('div'); last.className = 'c-text'; body.appendChild(last); }
          last.innerHTML = renderMD((last.textContent||'') + d.content);
          body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'tool'){
          status.textContent = '执行'; status.style.color = '#3b82f6';
          const tl = document.createElement('div'); tl.className = 'c-tool';
          tl.textContent = '... ' + d.tool; body.appendChild(tl);
          body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'result'){
          const rl = document.createElement('div');
          rl.className = 'c-tool ' + (d.success?'ok':'err');
          rl.textContent = (d.success?'OK ':'FAIL ') + (d.tool||'') + ': ' + (d.text||'').slice(0,80);
          body.appendChild(rl); body.scrollTop = body.scrollHeight;
        }
        else if(d.type === 'done'){
          status.textContent = d.success ? '完成' : '失败';
          status.style.color = d.success ? '#065f46' : '#991b1b';
        }
      }
      scrollDown();
    }
  }

  function scrollDown() { cleanupEmptyMsgs(); const m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; }
  function cleanupEmptyMsgs() { document.querySelectorAll('.msg').forEach(el=>{ const txt=el.querySelector('.msg-text'); if(txt && !txt.innerHTML.trim()) el.remove(); }); }
  async function onKey(e) {
    var isEnter = e.key === 'Enter' || e.keyCode === 13;
    if(isEnter && !e.shiftKey && !e.isComposing){
      e.preventDefault();
      await send();
    }
  }

  async function send() {
    const input = document.getElementById('input'); const text = input.value.trim();
    if(!text) return;
    if(!ws || !wsReady || ws.readyState!==WebSocket.OPEN){ toast('连接已断开，正在重连...'); connect(); return; }
    // 自动创建本地会话
    if(!activeSid){ saveCurrentMessages(); var id='ds_'+Date.now(); sessions.push({id:id,title:text.slice(0,30),status:'running',messages:[],serverSid:''}); saveSessions(); activeSid=id; renderSidebar(); }
    // 获取服务器会话 ID
    if(!sessionId){
      try{ var r=await fetch('/api/sessions',{method:'POST'}); var d=await r.json(); sessionId=d.session_id; var s=getSession(activeSid); if(s){s.serverSid=sessionId; saveSessions();} }catch(e){ toast('创建会话失败'); return; }
    }
    // 显示用户消息
    document.getElementById('welcome')?.remove();
    var div=document.createElement('div'); div.className='msg user';
    div.innerHTML='<div class="msg-body"><div class="msg-text">'+renderMD(text)+'</div></div>';
    document.getElementById('msgList').appendChild(div); scrollDown();
    input.value=''; input.style.height='auto';
    document.getElementById('sendBtn').disabled=true;
    ws.send(JSON.stringify({message:text,session_id:sessionId}));
    updateSessionStatus(activeSid, 'running');
  }

  function continueStage() {
    const awaitMsg = document.getElementById('awaitConfirmMsg');
    if(awaitMsg) awaitMsg.remove();
    if(ws && wsReady){
      ws.send(JSON.stringify({message:'继续',session_id:sessionId,action:'continue_stage'}));
    }
    document.getElementById('sendBtn').disabled = false;
  }


  async function newChat() {
    saveCurrentMessages();
    var id = 'ds_'+Date.now();
    sessions.push({id:id, title:'新对话', status:'done', messages:[], serverSid:''});
    saveSessions();
    activeSid = id;
    document.getElementById('msgList').innerHTML = '<div class="welcome" id="welcome"><div class="welcome-icon">&#9670;</div><h2>有什么我可以帮你的？</h2><p style="color:var(--text-dim)">我可以操控电脑、写代码、查资料、管理项目</p></div>';
    simpleMsg = null; agentCards = {}; agentsRow = null;
    sessionId = '';
    renderSidebar();
    document.getElementById('sendBtn').disabled = false;
  }
  document.getElementById('input').addEventListener('input',function(){ this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight,140)+'px'; });
  init();
